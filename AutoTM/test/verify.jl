@testset "Testing Lots of things" begin
    localcache = joinpath(@__DIR__, "cache", "fuzz.jls")

    backend = AutoTM.Backend("CPU")
    cache = AutoTM.Profiler.CPUKernelCache(localcache)

    # Queue up a collection of models and optimizers.
    batchsize = 16 
    fns = [
        () -> AutoTM.Zoo.resnet_training(AutoTM.Zoo.Resnet50(), batchsize),
        () -> AutoTM.Zoo.vgg_training(AutoTM.Zoo.Vgg19(), batchsize),
        () -> AutoTM.Zoo.inception_v4_training(batchsize),
        # TODO: Finish profiling to add this test.
        #() -> AutoTM.Zoo.densenet_training(batchsize)
    ]

    # Go through the static and synchronous optimizers.
    #
    # Just do one where we know we'll get move nodes / a mix of DRAM and PMM
    # Keep the total number down to facilitate quicker testing.
    optimizers = [
        AutoTM.Optimizer.Static(1 // 1),
        AutoTM.Optimizer.Synchronous(1 // 1),
    ]

    for (i, f) in enumerate(fns)
        for (j, opt) in enumerate(optimizers)
            printstyled("Function $i -- Optimizer $j\n"; color = :green)
            @test AutoTM.Verifier.verify(backend, f, opt, cache)
        end
    end
end
