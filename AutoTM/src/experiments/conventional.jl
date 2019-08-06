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

function single_kernel_profile()
    fns = (
        #test_vgg(), 
        conventional_inception(),
        conventional_resnet(),
        conventional_vgg(),
    )

    backend = nGraph.Backend("CPU")
    for f in fns
        fex = actualize(backend, f)
        cache = Profiler.CPUKernelCache(SINGLE_KERNEL_PATH)
        Profiler.profile(fex; cache = cache, kernel_tiling = TILING_FACTOR[SINGLE_KERNEL_PATH])
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
        test_vgg(),
    )

    ratios = common_ratios()
    formulations = (
        "static",
        "synchronous",
    )

    caches = [
        SINGLE_KERNEL_PATH,
        MULTIPLE_KERNEL_PATH
    ]

    Visualizer.pgf_error_plot(fns, ratios, caches; formulations = formulations)
end
