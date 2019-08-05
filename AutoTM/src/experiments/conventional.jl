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

const SINGLE_KERNEL_PATH = joinpath(DATADIR, "caches", "single_cpu_profile.jls")
const MULTIPLE_KERNEL_PATH = joinpath(DATADIR, "caches", "multiple_cpu_profile.jls")

function single_kernel_profile()
    fns = (
        #conventional_vgg(),
        test_vgg(), 
    )

    backend = nGraph.Backend("CPU")
    for f in fns
        fex = actualize(backend, f)
        cache = Profiler.CPUKernelCache(SINGLE_KERNEL_PATH)
        Profiler.profile(fex; cache = cache, kernel_tiling = 1)
    end
end

function multi_kernel_profile()
    fns = (
        #conventional_vgg(),
        test_vgg(), 
    )

    backend = nGraph.Backend("CPU")
    for f in fns
        fex = actualize(backend, f)
        cache = Profiler.CPUKernelCache(MULTIPLE_KERNEL_PATH)
        Profiler.profile(fex; cache = cache, kernel_tiling = 4)
    end
end
