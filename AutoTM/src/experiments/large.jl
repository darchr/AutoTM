large_inception() = Inception_v4(6144)
large_vgg() = Vgg416(320)

function kernel_profile(fns; recache = false)
    backend = nGraph.Backend("CPU")
    dummy_opt = Optimizer.Static(1 // 0)

    for f in wrap(fns)
        Optimizer.factory(backend, f, dummy_opt; 
            cache = Profiler.CPUKernelCache(SINGLE_KERNEL_PATH),
            just_profile = true, 
            recache = recache
        )
    end
end
