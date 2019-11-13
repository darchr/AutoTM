const WORKLOAD_MAP = Dict(
    "test_vgg" => AutoTM.Experiments.test_vgg(),
    "large_vgg" => AutoTM.Experiments.large_vgg(),
    "large_inception" => AutoTM.Experiments.large_inception(),
    "large_resnet" => AutoTM.Experiments.large_resnet(),
    "large_densenet" => AutoTM.Experiments.large_densenet(),
)

workloads() = keys(WOAKLOAD_MAP)

const OPTIMIZER_MAP = Dict(
    "1lm" => AutoTM.Optimizer.Synchronous(185_000_000_000),
    "2lm" => AutoTM.Optimizer.Optimizer2LM(),
)

struct AutoTMParams
    workload::String
    cache::Any
    # 1LM or 2LM
    mode::String
    # Bool flag to use scratchpad
    use_2lm_scratchpad::Bool
    pipe::Any
end

getworkload(A::AutoTMParams) = WORKLOAD_MAP[A.workload]
getoptimizer(A::AutoTMParams) = OPTIMIZER_MAP[A.mode]

maybeunwrap(x) = x
maybeunwrap(x::Tuple) = first(x)

# Hoist into a function so GC works correctly
function run(backend, params::AutoTMParams, measurements)
    # Get workload and optimizer.
    f = getworkload(params)
    opt = getoptimizer(params)

    # Build up a list of optional keyword arguments.
    kw = []
    if params.use_2lm_scratchpad
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
