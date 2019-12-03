#####
##### Run in 1LM
#####

test_vgg() = AutoTM.Experiments.test_vgg()

conventional_inception()    = AutoTM.Experiments.conventional_inception()
conventional_resnet()       = AutoTM.Experiments.conventional_resnet()
conventional_vgg()          = AutoTM.Experiments.conventional_vgg()
conventional_densenet()     = AutoTM.Experiments.conventional_densenet()
#conventional_transformer()  = AutoTM.Experiments.conventional_transformer()

common_ratios() = [
    1 // 0,
    8 // 1,
    4 // 1,
    1 // 1,
    0 // 1,
]

conventional_functions() = [
    conventional_inception(),
    conventional_resnet(),
    conventional_vgg(),
    conventional_densenet(),
]

#####
##### Standard Routine
#####

"""
    run_conventional(fn, optimizers, ratios)

Run experiment for a conventional workload.

Argument Description
--------------------

* `fn`: The network to run. Options are
    - `test_vgg()`
    - `conventional_vgg()`
    - `conventional_inception()`
    - `conventional_resnet()`
    - `conventional_densenet()`

* `optimizers`: Iteratable of optimizers to use. Options:
    - `AutoTM.Optimizers.Static`
    - `AutoTM.Optimizers.Synchronous`
    - `AutoTM.Optimizers.Numa`

* `ratios`: Iterable of ratios (`Rational{Int}`) of ratios of PMM and DRAM to use.
    Defaults to `conventional_ratios() = [1 // 0, 8 // 1, 4 // 1, 1 // 1, 0 // 1]`

Results will be saved to `Benchmarker/data/cpu`.
"""
function run_conventional(fn, optimizers, ratios = common_ratios())
    ratios = common_ratios()

    opt = Iterators.flatten(o.(ratios) for o in optimizers)
    cache = AutoTM.Experiments.CPU_CACHE
    execute(fn, opt, cache, nGraph.Backend("CPU"))
end

############################################################################################
# Plots
############################################################################################

#####
##### Front Plot
#####

function plot_front()
    f = conventional_inception()
    ratio = 4 // 1
    cache = AutoTM.Experiments.CPU_CACHE
    formulations = ("numa", "synchronous",)

    plot_front(f, ratio, cache; formulations = formulations)
end

#####
##### PMM Speedup Plot
#####

function plot_speedup(
        models = conventional_functions();
        formulations = ("numa", "static", "synchronous"),
        cache = AutoTM.Experiments.CPU_CACHE
    )
    ratios = common_ratios();

    # Get rid of the all PMEM and all DRAM case
    deleteat!(ratios, findfirst(isequal(0 // 1), ratios))
    deleteat!(ratios, findfirst(isequal(1 // 0), ratios))

    pgf_speedup(
        models,
        ratios,
        cache;
        formulations = formulations,
    )
end

#####
##### Error Plot
#####

function plot_conventional_error(;
        fns = [
            conventional_inception(),
            conventional_resnet(),
            conventional_vgg(),
            conventional_densenet()
        ],
        ratios = common_ratios(),
        formulations = ("static", "synchronous")
    )

    caches = [
        AutoTM.Experiments.CPU_CACHE
    ]

    suffix = nothing
    return pgf_error_plot(fns, ratios, caches; formulations = formulations, suffix = suffix)
end

#####
##### Cost Plot
#####

function plot_costs()
    pairs = [
        conventional_vgg() => "synchronous",
        conventional_densenet() => "synchronous",
        conventional_resnet() => "synchronous",
        conventional_inception() => "synchronous",
    ]

    ratios = common_ratios();

    # Get rid of the all PMEM and all DRAM case
    deleteat!(ratios, findfirst(isequal(0 // 1), ratios))
    cache = AutoTM.Experiments.CPU_CACHE

    pgf_cost(pairs, ratios, cache; cost_ratio = 2.1)
end

#####
##### Case Study - Inception
#####

function case_study_ratios()
    # Start out with the ratios of the whole memory we want to devote to DRAM
    ratios = [ 1 // i for i in 1:10 ]

    # Add some more fractions for the higher end of DRAM and for the case of all PMEM
    push!(ratios,
        1 // 100,
        1 // 80,
        1 // 60,
        1 // 40,
        1 // 20,
        2 // 3,
        3 // 4,
        0 // 1
    )
    sort!(ratios)

    # Convert these ratios into the PMEM // DRAM ratios
    pmem_to_dram_ratios = (one(eltype(ratios)) .- ratios) ./ ratios
    return pmem_to_dram_ratios
end

function inception_case_study()
    f = conventional_inception()

    ratios = case_study_ratios()

    optimizers = Iterators.flatten((
        [Optimizer.Static(r) for r in ratios],
        [Optimizer.Synchronous(r) for r in ratios],
    ))

    cache = SINGLE_KERNEL_PATH
    suffix = "study"

    # Run the workload - don't perform the ratio search because that's not as important
    # for this experiment
    execute(f, optimizers, cache, nGraph.Backend("CPU"), suffix; search_ratio = false)
end

function inception_case_study_plots()
    f = conventional_inception()
    cache = AutoTM.Experiments.CPU_CACHE
    suffix = "study"
    formulations = (
        "static",
        "synchronous",
    )

    # Performance line graph
    a = pgf_plot_performance(f, cache, suffix;
        file = joinpath(FIGDIR, "inception_perf.pdf"),
        formulations = ("static", "synchronous"),
    )

    # Input/output tensor graph
    b = pgf_io_plot(f, cache, suffix;
        file = joinpath(FIGDIR, "inception_io.pdf"),
        formulations = formulations,
    )

    # Movement statistics
    c = pgf_movement_plot(f, cache, suffix;
        file = joinpath(FIGDIR, "inception_movement.pdf"),
        formulation = "synchronous"
    )

    return (
        performance = a,
        io = b,
        movement = c,
    )
end

#####
##### Large Models
#####

large_inception() = AutoTM.Experiments.large_inception()
large_resnet() = AutoTM.Experiments.large_resnet()
large_densenet() = AutoTM.Experiments.large_densenet()
large_vgg() = AutoTM.Experiments.large_vgg()

"""
    kernel_profile(fns; recache = false)

Perform kernel profiling for each function in `fns`.
Possible values for argument `fns` are:
- `test_vgg()`
- `conventional_vgg()`
- `conventional_inception()`
- `conventional_resnet()`
- `conventional_densenet()`
- `large_vgg()`
- `large_inception()`
- `large_resnet()`
- `large_densenet()`

If `recache = true`, delete previously profiled kernels for each function.
"""
function kernel_profile(fns; recache = false)
    backend = nGraph.Backend("CPU")
    dummy_opt = AutoTM.Optimizer.Static(0)

    for f in wrap(fns)
        AutoTM.Optimizer.factory(
            backend,
            f,
            dummy_opt;
            cache = AutoTM.Profiler.CPUKernelCache(AutoTM.Experiments.CPU_CACHE),
            just_profile = true,
            recache = recache
        )
    end
end

# Default the limit to 185 GB, slightly smaller than the 192 GB DRAM size.
function run_large(f, opt; limit = 185_000_000_000)
    optimizer = opt(limit)
    backend = nGraph.Backend("CPU")
    cache = SINGLE_KERNEL_PATH

    execute(f, optimizer, cache, nGraph.Backend("CPU"); skip_base_check = true, defrag = true)
end

#####
##### 2LM
#####

# Just execute the function directly - nothing too fancy here
function run_2lm(fn)
    opt = Optimizer.Optimizer2LM()
    cache = "nocache"
    backend = nGraph.Backend("CPU")

    savefile = canonical_path(fn, opt, cache, backend)
    stats = ispath(savefile) ? deserialize(savefile) : AutoTM.Profiler._base_stats()
    fex = Optimizer.factory(backend, fn, opt)
    runtime = gettime(fex)

    run = Dict(
        :actual_runtime => runtime,
    )
    push!(stats.runs, run)
    push!(stats.runs, run)
    serialize(savefile, stats)
end

# Generate the plot for the large workloads
function plot_large()
    fns = (large_vgg(), large_inception(), large_resnet(), large_densenet())
    cache = AutoTM.Experiments.CPU_CACHE
    cache_2lm = "nocache"
    formulations = ("static", "synchronous")

    pgf_large_performance(fns, cache, cache_2lm; formulations = formulations)
end

#####
##### GPU Benchmarks
#####

const GPU_MAX_MEMORY = 11_000_000_000

# Got this number from looking at nvidia-smi after all the GPU initialization code
# in ngraph runs.
#
# Probably worth double-checking
const GPU_MEMORY_OVERHEAD = 561_000_000
const GPU_ADJUSTED_MEMORY = GPU_MAX_MEMORY - GPU_MEMORY_OVERHEAD

gpu_fns() = (
    AutoTM.Experiments.Inception_v4(64),
    AutoTM.Experiments.Inception_v4(128),
    AutoTM.Experiments.Inception_v4(256),
    AutoTM.Experiments.Resnet200(32),
    AutoTM.Experiments.Resnet200(64),
    AutoTM.Experiments.Resnet200(128),
    AutoTM.Experiments.DenseNet(32),
    AutoTM.Experiments.DenseNet(64),
    AutoTM.Experiments.DenseNet(128),
    AutoTM.Experiments.Vgg19(64),
    AutoTM.Experiments.Vgg19(128),
)

function gpu_profile(; recache = false, allow_alloc_fail = false)
    fns = gpu_fns()
    opt = AutoTM.Optimizer.Synchronous(GPU_ADJUSTED_MEMORY)
    cache = GPU_CACHE
    backend = nGraph.Backend("GPU")

    for f in fns
        @show name(f)
        try
            execute(f, opt, cache, backend;
                just_profile = true,
                skip_base_check = true,
                allow_alloc_fail = allow_alloc_fail,
                recache = recache,
            )
        catch e
            isa(e, AutoTM.Profiler.CompilerExit) || rethrow(e)
        end
    end
end

function gpu_go(i)
    fns = gpu_fns()[i]
    limit = GPU_ADJUSTED_MEMORY

    optimizers = (
        Optimizer.Synchronous(limit),
        Optimizer.Asynchronous(limit),
    )

    cache = GPU_CACHE
    backend = nGraph.Backend("GPU")

    execute(fns, optimizers, cache, backend; adjust_io = true)
end

plot_gpu_performance() = pgf_gpu_performance_plot(gpu_fns(), AutoTM.Experiments.GPU_CACHE)
