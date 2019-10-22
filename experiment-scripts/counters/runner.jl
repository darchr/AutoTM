# Instructions for use:
#
# Start julia with
#     sudo ~/julia-1.2.0/bin/julia
#
# -- We must not run under "sudo", but `counters.jl` must run under "sudo" in order to
# sample the counters.
#
# The script will run through a collection of AutoTM experiments to run, and before running
# will invoke a job on the second process to start sampling.
#
# The master process will write to a Named Pipe when the ngraph function has completed -
# at which point the secondary process will stop sampling and serialize its sampled data
# to a filepath provided by the master worker.


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
using ArgParse
using Sockets

const WORKLOADS = [
    "test_vgg",
    "large_vgg",
    "large_resnet",
    "large_densenet",
    "large_inception",
]

# Entrypoint argument parsing
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--mode"
            required = true
            arg_type = String
            range_tester = x -> in(x, ("1lm", "2lm"))
            help = "Select between [1lm] or [2lm]"

        "--workload"
            required = true
            arg_type = String
            nargs = '+'
            range_tester = x -> in(x, WORKLOADS)
            help = """
            Select the workload to run. Options: $(join(WORKLOADS, ", ", ", and "))
            Multiple Options Supported.
            """

        "--counter_type"
            arg_type = String
            default = ["rw"]
            nargs = '+'
            # Make sure argument is one of the tag groups we know about.
            range_tester = x -> in(x, ("rw", "tags", "queues", "insert-check"))
            help = """
            Select which sets of counters to use.
            [rw]: Use DRAM/PMEM read/write counters
            [tags]: Return dirty/clean miss count and hit count.
            [queue]: Return the read and write queue occupancy for PMM.
            [insert-check]: Check semantic difference of read/write insert vs read/write
            command counters.
            """

        "--use_2lm_scratchpad"
            action = :store_true
            help = "Use a dedicated scratchpad for short lived tensors in 2LM mode."

        "--sampletime"
            arg_type = Int
            default = 1
            help = "The number of seconds between sampling the hardware counters."
    end

    return parse_args(s)
end

#####
##### Here's the main sampling script.
#####

function get_optimizer(mode)
    if mode == "1lm"
        opt = AutoTM.Optimizer.Synchronous(185_000_000_000)
    elseif mode == "2lm"
        opt = AutoTM.Optimizer.Optimizer2LM()
    else
        throw(ArgumentError("Unknown mode $mode"))
    end
    return opt
end

function get_workload(workload)
    if workload == "test_vgg"
        f = AutoTM.Experiments.test_vgg()
    elseif workload == "large_vgg"
        f = AutoTM.Experiments.large_vgg()
    elseif workload == "large_inception"
        f = AutoTM.Experiments.large_inception()
    elseif workload == "large_resnet"
        f = AutoTM.Experiments.large_resnet()
    elseif workload == "large_densenet"
        f = AutoTM.Experiments.large_densenet()
    else
        throw(ArgumentError("Unknown workload $workload"))
    end
    return f
end

function make_filename(workload, mode, counter_type, use_2lm_scratchpad::Bool)
    # Join the first three arguments together.
    # If we're in 2LM mode, check to see if we're using the scratchpad. If so, also
    # indicate that in the path name.
    str = join((workload, mode, counter_type), "_")
    if mode == "2lm" && use_2lm_scratchpad
        str = str * "_scratchpad"
    end

    # Append the extension.
    return "data/$(str).jls"
end

maybeunwrap(x) = x
maybeunwrap(x::Tuple) = first(x)

# Hoist into a function so GC works correctly
function run(backend, workload, cache, pipe, parsed_args)
    # Get workload and optimizer.
    f = get_workload(workload)
    opt = get_optimizer(parsed_args["mode"])

    # Build up a list of optional keyword arguments.
    kw = []
    if parsed_args["use_2lm_scratchpad"]
        push!(kw, :use_scratchpad => true)
    end

    # Instantiate the ngraph function to run - go through `factory` so the node affinity
    # optimization pass runs

    fex = AutoTM.Optimizer.factory(
        backend,
        f,
        opt;
        cache = cache,
        kw...
    )

    # Sometimes, we might get a tuple back from factory - unpack it so we just get the
    # FlusExecutable.
    fex = maybeunwrap(fex)

    # Run the function once to warm it up.
    @time fex()

    # Reuse the initial compilation for all counter sets
    for counter_set in parsed_args["counter_type"]
        println("Counter Set: $counter_set")
        filepath = make_filename(
            workload,
            parsed_args["mode"],
            counter_set,
            parsed_args["use_2lm_scratchpad"]
        )

        # Configure `counters.jl`
        println(pipe, "filepath $filepath")
        println(pipe, "counters $(counter_set)")

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
    return nothing
end

# Main function.
function main()
    # First - parse commandline arguments.
    parsed_args = parse_commandline()

    # Setup some static data structures
    backend = AutoTM.Backend("CPU")
    pipe = connect(pipe_name)
    cache = AutoTM.Profiler.CPUKernelCache(AutoTM.Experiments.SINGLE_KERNEL_PATH)

    workloads = parsed_args["workload"]
    println(pipe, "sampletime = $(parsed_args["sampletime"])")

    for workload in workloads
        println("Running $workload")

        # Run
        run(backend, workload, cache, pipe, parsed_args)
    end

    return nothing
end

# Invoke the main function.
main()
