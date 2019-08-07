#####
##### Run in 1LM
#####

# Small network for debugging and such
test_vgg() = Vgg19(16)

conventional_inception() = Inception_v4(1024)
conventional_resnet() = Resnet200(512)
conventional_vgg() = Vgg19(2048)
conventional_densenet() = DenseNet(512)
conventional_transformer() = Transformer(512, 200)

common_ratios() = [
    1 // 0,
    8 // 1,
    4 // 1,
    #2 // 1,
    1 // 1,
    #1 // 2,
    #1 // 4,
    0 // 1,
]

conventional_functions() = [
    conventional_inception(),
    conventional_resnet(),
    conventional_vgg(),
    conventional_densenet(),
    conventional_transformer(),
]

function single_kernel_profile(; recache = false)
    fns = (
        #test_vgg(),
        #conventional_inception(),
        #conventional_resnet(),
        #conventional_vgg(),
        conventional_densenet(),
    )

    backend = nGraph.Backend("CPU")
    for f in fns
        fex = actualize(backend, f)
        cache = Profiler.CPUKernelCache(SINGLE_KERNEL_PATH)
        Profiler.profile(fex;
            recache = recache,
            cache = cache,
            kernel_tiling = TILING_FACTOR[SINGLE_KERNEL_PATH]
        )
    end
end

function multi_kernel_profile(; recache = false)
    fns = (
        #conventional_vgg(),
        #test_vgg(),
        conventional_inception(),
        conventional_resnet(),
        conventional_vgg(),
    )

    backend = nGraph.Backend("CPU")
    for f in fns
        fex = actualize(backend, f)
        cache = Profiler.CPUKernelCache(MULTIPLE_KERNEL_PATH)
        Profiler.profile(fex; cache = cache, kernel_tiling = TILING_FACTOR[MULTIPLE_KERNEL_PATH])
    end
end

#####
##### Standard Routine
#####
function run_conventional()
    fns = (
        #test_vgg(),
        #conventional_inception(),
        #conventional_resnet(),
        conventional_vgg(),
    )

    ratios = common_ratios()

    optimizers = Iterators.flatten((
        #[Optimizer.Static(r) for r in ratios],
        [Optimizer.Synchronous(r) for r in ratios],
    ))

    caches = [
        SINGLE_KERNEL_PATH,
        #MULTIPLE_KERNEL_PATH
    ]

    execute(fns, optimizers, caches, nGraph.Backend("CPU"))
end

#####
##### PMM Speedup Plot
#####

function plot_speedup(
        model;
        formulations = ("static", "synchronous"),
        cache = SINGLE_KERNEL_PATH,
    )
    ratios = common_ratios();

    # Get rid of the all PMEM and all DRAM case
    deleteat!(ratios, findfirst(isequal(0 // 1), ratios))
    deleteat!(ratios, findfirst(isequal(1 // 0), ratios))

    Visualizer.pgf_speedup(
        model,
        ratios,
        cache;
        formulations = formulations,
    )
end

#####
##### Error Plot
#####

function plot_conventional_error()
    fns = (
        #test_vgg(),
        conventional_inception(),
        conventional_resnet(),
        conventional_vgg(),
    )

    ratios = common_ratios()
    # Start out with the ratios of the whole memory we want to devote to DRAM
    #ratios = [ 1 // i for i in 1:10 ]

    ## Add some more fractions for the higher end of DRAM and for the case of all PMEM
    #push!(ratios, 2 // 3, 3 // 4, 0 // 1)
    #sort!(ratios)

    ## Convert these ratios into the PMEM // DRAM ratios
    #pmem_to_dram_ratios = (one(eltype(ratios)) .- ratios) ./ ratios
    #ratios = pmem_to_dram_ratios
    formulations = (
        "static",
        "synchronous",
    )

    caches = [
        SINGLE_KERNEL_PATH,
        #MULTIPLE_KERNEL_PATH
    ]

    #suffix = "study"
    suffix = nothing

    Visualizer.pgf_error_plot(fns, ratios, caches; formulations = formulations, suffix = suffix)
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
        #[Optimizer.Static(r) for r in pmem_to_dram_ratios],
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
    cache = SINGLE_KERNEL_PATH
    suffix = "study"
    formulations = (
        #"static",
        "synchronous",
    )

    # Performance line graph
    # Visualizer.pgf_plot_performance(f, cache;
    #     file = "inception_perf.tex",
    #     formulations = ("static", "synchronous"),
    # )

    # Input/output tensor graph
    Visualizer.pgf_io_plot(f, cache, suffix;
        file = "inception_io.tex",
        formulations = formulations,
    )

    # Movement statistics
    Visualizer.pgf_movement_plot(f, cache, suffix;
        file = "inception_movement.tex",
        formulation = "synchronous"
    )
end

