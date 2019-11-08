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
include("init.jl")

using Dates
using AutoTM
using nGraph
using ArgParse
using Sockets
using Serialization

# Like serialize, but will also make a directory if needed.
function save(file, x)
    dir = dirname(file)
    isdir(dir) || mkpath(dir)
    serialize(file, x)
    return nothing
end

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
            nargs = '*'
            # Make sure argument is one of the tag groups we know about.
            range_tester = x -> in(x, ("rw", "tags", "queues", "insert-check"))
            help = """
            Select which sets of counters to use.
            If argument is not provided, counters will not be used.
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

        "--save_kerneltimes"
            action = :store_true
            help = """
            Run the compiled function. Save a `AutoTM.Profiler.FunctioData` object as well
            as profiled kernel times in the `serialized` folder for generating heap plots.

            This option invalidates any counters.
            """
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

function make_filename(workload, mode, counter_type, use_2lm_scratchpad::Bool; prefix = "data")
    # Join the first three arguments together.
    # If we're in 2LM mode, check to see if we're using the scratchpad. If so, also
    # indicate that in the path name.
    str = join((workload, mode, counter_type), "_")
    if mode == "2lm" && use_2lm_scratchpad
        str = str * "_scratchpad"
    end

    # Append the extension.
    return "$(prefix)/$(str).jls"
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

        # Signal worker process that we're done by writing to the NamedPipe
        println(pipe, "stop")
        println("Successfully Ran Worker Process")
    end
    return nothing
end

function saverun(backend, workload, cache, parsed_args)
    f = get_workload(workload)
    opt = get_optimizer(parsed_args["mode"])

    # Build up a list of optional keyword arguments.
    kw = []
    if parsed_args["use_2lm_scratchpad"]
        push!(kw, :use_scratchpad => true)
    end

    fex = AutoTM.Optimizer.factory(
        backend,
        f,
        opt;
        cache = cache,
        kw...
    ) |> maybeunwrap

    # Run twice - we'll get timing data from the second iteration.
    @time fex()

    # Reset the performance counters so they aren't "contaminated" from the original run.
    #
    # The performance counters accumulate across runs, so if we don't do this, we will
    # essentially be averaging the runtime from the second run with that of the first run.
    nGraph.reset_counters(fex.ex)
    @time fex()

    # Create a `FunctionData` object from the nGraph function.
    # Couple it with the performance metrics.
    function_data = AutoTM.Profiler.FunctionData(fex.ex.ngraph_function, backend)
    times = nGraph.get_performance(fex.ex)

    println("Sum of Kernel Run Times: $(sum(values(times)))")

    # Record metadata about each tensor that will be used for plot generation.
    tensor_records = map(collect(AutoTM.Profiler.tensors(function_data))) do tensor
        users = AutoTM.Profiler.users(tensor)
        return Dict(
             "name" => nGraph.name(tensor),
             "users" => nGraph.name.(users),
             "user_indices" => getproperty.(users, :index),
             "sizeof" => sizeof(tensor),
             "offset" => Int(AutoTM.Profiler.getoffset(tensor)),
        )
    end

    # Collect metadata on each node.
    node_records = map(AutoTM.Profiler.nodes(function_data)) do node
        outputs = nGraph.name.(AutoTM.Profiler.outputs(node))
        inputs = nGraph.name.(AutoTM.Profiler.inputs(node))
        time = get(times, nGraph.name(node), 0)
        delete!(times, nGraph.name(node))
        return Dict(
            "name" => nGraph.name(node),
            "inputs" => inputs,
            "outputs" => outputs,
            "time" => time,
        )
    end

    @assert isempty(times)

    # Combine together to the final record that will be saved.
    record = Dict(
        "tensors" => tensor_records,
        "nodes" => node_records,
    )

    file = make_filename(
        workload,
        parsed_args["mode"],
        "",
        parsed_args["use_2lm_scratchpad"];
        prefix = joinpath(@__DIR__, "serialized")
    )
    save(file, record)
    return nothing
end

# Main function.
function main()
    # First - parse commandline arguments.
    parsed_args = parse_commandline()

    # Setup some static data structures
    backend = AutoTM.Backend("CPU")
    cache = AutoTM.Profiler.CPUKernelCache(AutoTM.Experiments.CPU_CACHE)

    workloads = parsed_args["workload"]

    # Some dummy detection logic
    if get(parsed_args, "save_kerneltimes", false)
        for workload in workloads
            saverun(backend, workload, cache, parsed_args)
        end
    else
        # We aren't saving kernel times - make sure that we recieved at least one counter.
        if isempty(get(parsed_args, "counter_type", []))
            println("Need to specify at least one counter_type")
            return nothing
        end

        if !ispath(pipe_name)
            println("Make sure `counters.jl` is running, and try again.")
            return nothing
        end

        pipe = connect(pipe_name)
        println(pipe, "sampletime = $(parsed_args["sampletime"])")

        for workload in workloads
            println("Running $workload")
            run(backend, workload, cache, pipe, parsed_args)
        end
    end

    return nothing
end

# Invoke the main function.
main()
