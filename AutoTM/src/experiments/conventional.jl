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
        conventional_vgg(),
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
        conventional_inception(),
        conventional_resnet(),
        conventional_vgg(),
    )

    ratios = common_ratios()

    optimizers = Iterators.flatten((
        [Optimizer.Static(r) for r in ratios],
        [Optimizer.Synchronous(r) for r in ratios],
    ))

    caches = [
        SINGLE_KERNEL_PATH,
        MULTIPLE_KERNEL_PATH
    ]

    execute(fns, optimizers, caches, nGraph.Backend("CPU"))
end

function plot_conventional_error()
    fns = (
        #test_vgg(),
        conventional_inception(),
        #conventional_resnet(),
        #conventional_vgg(),
    )

    ratios = common_ratios()
    formulations = (
        "static",
        "synchronous",
    )

    caches = [
        SINGLE_KERNEL_PATH,
        #MULTIPLE_KERNEL_PATH
    ]

    Visualizer.pgf_error_plot(fns, ratios, caches; formulations = formulations)
end

#####
##### Case Study - Inception
#####

function inception_case_study()
    f = conventional_inception()

    # Start out with the ratios of the whole memory we want to devote to DRAM
    ratios = [ 1 // i for i in 1:10 ]

    # Add some more fractions for the higher end of DRAM and for the case of all PMEM
    push!(ratios, 2 // 3, 3 // 4, 0 // 1) 
    sort!(ratios)

    # Convert these ratios into the PMEM // DRAM ratios
    pmem_to_dram_ratios = (one(eltype(ratios)) .- ratios) ./ ratios
    optimizers = Iterators.flatten((
        [Optimizer.Static(r) for r in pmem_to_dram_ratios],
        [Optimizer.Synchronous(r) for r in pmem_to_dram_ratios],
    ))

    cache = SINGLE_KERNEL_PATH

    # Run the workload - don't perform the ratio search because that's not as important
    # for this experiment
    execute(f, optimizers, cache, nGraph.Backend("CPU"); search_ratio = false)
end
