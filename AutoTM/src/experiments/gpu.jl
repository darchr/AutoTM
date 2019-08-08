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
    Inception_v4(64),
    Inception_v4(128),
    Inception_v4(256),
    Resnet200(32),
    Resnet200(64),
    Resnet200(128),
    DenseNet(32),
    DenseNet(64),
    DenseNet(128),
    Vgg19(64),
    Vgg19(128),
    Vgg19(256),
)

function gpu_profile(; recache = false)
    fns = gpu_fns()
    opt = Optimizer.Synchronous(GPU_ADJUSTED_MEMORY)
    cache = GPU_CACHE
    backend = nGraph.Backend("GPU")

    for f in fns
        @show name(f)
        try
            execute(f, opt, cache, backend; 
                just_profile = true, 
                skip_base_check = true,
                allow_alloc_fail = false,
                recache = recache,
            )
        catch e
            isa(e, Profiler.CompilerExit) || rethrow(e)
        end
    end
end

function gpu_go()
    fns = gpu_fns()
    limit = GPU_ADJUSTED_MEMORY

    optimizers = (
        #Optimizer.Synchronous(limit),
        Optimizer.Asynchronous(limit),
    )

    cache = GPU_CACHE
    backend = nGraph.Backend("GPU")

    execute(fns, optimizers, cache, backend; adjust_io = true)
end

plot_gpu_performance() = Runner.pgf_gpu_performance_plot(gpu_fns())
