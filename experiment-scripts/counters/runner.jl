# Instructions for use:
#
# Start julia with
#     sudo ~/julia-1.2.0/bin/julia
#
# -- We must run under `sudo` to have access to the performance counters.
#
# The script will run through a collection of AutoTM experiments to run, and before running
# will invoke a job on the second process to start sampling.
#
# The master process will write to a Named Pipe when the ngraph function has completed -
# at which point the secondary process will stop sampling and serialize its sampled data
# to a filepath provided by the master worker.

#####
##### Setup
#####

# Select which set of tests to run
#mode = "test"
mode = "2lm"
#mode = "1lm"

# How often to sample counters
sampletime = 1      # Seconds

# Which set of counters to use
# - `rw`: Collect read/write counters for DRAM and PMM
# - `tags`: Collect tag check metrics and PMM write queue occupancy
counters = "rw"
#counters = "tags"

# Flag on if to use a `scratchpad` during 2LM execution.
use_2lm_scratchpad = true

# The name of the Pipe to use for communication with `counter.jl`
pipe_name = "counter_pipe"

#####
##### Code Loading
#####

# Activate the AutoTM environment, which will load all the correct versions of dependencies
using Pkg
Pkg.activate("../../AutoTM")
using Dates
using AutoTM
using Sockets

#####
##### Here's the main sampling script.
#####

if mode == "test"
    trials = [
        (AutoTM.Experiments.conventional_inception(), "test_vgg.jls")
    ]
    opt = AutoTM.Optimizer.Optimizer2LM()
    cache = AutoTM.Profiler.CPUKernelCache(AutoTM.Experiments.SINGLE_KERNEL_PATH)
elseif mode == "1lm"
    trials = [
        #(AutoTM.Experiments.large_inception(),  "inception_1lm.jls"),
        (AutoTM.Experiments.large_vgg(),        "vgg_1lm.jls"),
        (AutoTM.Experiments.large_resnet(),     "resnet_1lm.jls"),
        (AutoTM.Experiments.large_densenet(),   "densenet_1lm.jls"),
    ]
    opt = AutoTM.Optimizer.Synchronous(185_000_000_000)
    cache = AutoTM.Profiler.CPUKernelCache(AutoTM.Experiments.SINGLE_KERNEL_PATH)
elseif mode == "2lm"
    trials = [
        (AutoTM.Experiments.large_inception(),  "inception_scratchpad_2lm.jls"),
        (AutoTM.Experiments.large_vgg(),        "vgg_scratchpad_2lm.jls"),
        (AutoTM.Experiments.large_resnet(),     "resnet_scratchpad_2lm.jls"),
        (AutoTM.Experiments.large_densenet(),   "densenet_scratchpad_2lm.jls"),
    ]
    opt = AutoTM.Optimizer.Optimizer2LM()
    cache = nothing
else
    throw(error("Don't under stand mode $mode"))
end

maybeunwrap(x) = x
maybeunwrap(x::Tuple) = first(x)

# Hoist into a function so GC works correctly
function run(backend, f, opt, cache, pipe)

    # Instantiate the ngraph function to run - go through `factory` so the node affinity
    # optimization pass runs
    fex = AutoTM.Optimizer.factory(
        backend, 
        f, 
        opt; 
        cache = cache, 
        use_scratchpad = use_2lm_scratchpad
    )

    # Sometimes, we might get a tuple back from factory - unpack it so we just get the 
    # FlusExecutable.
    fex = maybeunwrap(fex)

    # Run the function once to warm it up.
    @time fex() 

    # Invoke the sampling subprocess
    println("Invoking Sampler")
    println(pipe, "start")

    # Sleep for 5 seconds to give the subprocess time to startup and invoke the 
    # FluxExecutable again.
    sleep(5)
    @time fex() 
    sleep(5)

    # Signal worker process that we're done by writing to the RemoteChannel
    println(pipe, "stop")
    println("Successfully Ran Worker Process")
end

# Try running things - see if it works
backend = AutoTM.Backend("CPU")

pipe = connect(pipe_name)

for trial in trials
    # Unpack trial struct
    f = trial[1]
    filepath = trial[2]

    # Configure `counters.jl`
    println(pipe, "sampletime $sampletime")
    println(pipe, "filepath $filepath")
    println(pipe, "counters $counters")

    # Run
    run(backend, f, opt, cache, pipe)
end
