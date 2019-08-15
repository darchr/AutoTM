large_inception() = Inception_v4(6144)      # 659 GB
large_vgg() = Vgg416(320)                   # 658 GB
large_resnet() = Resnet200(2560)            # 651 GB
large_densenet() = DenseNet(3072)           # 688 GB

# Set the DRAM size to use for the networks
const SIZE_DICT = Dict(
    name(large_inception()) => 185_000_000_000,
    name(large_vgg())       => 175_000_000_000,
)

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

function run_large(f, opt; limit = SIZE_DICT[name(f)])
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
    stats = ispath(savefile) ? deserialize(savefile) : Profiler._base_stats()
    fex = Optimizer.factory(backend, fn, opt)
    runtime = gettime(fex)

    run = Dict(
        :actual_runtime => runtime,
    )
    push!(stats.runs, run)
    push!(stats.runs, run)
    serialize(savefile, stats)
end

#####
##### Large Plot
#####

function plot_large()
    fns = (large_vgg(), large_inception())
    cache = SINGLE_KERNEL_PATH
    cache_2lm = "nocache"
    formulations = ("static", "synchronous")

    Visualizer.pgf_large_performance(fns, cache, cache_2lm; formulations = formulations)
end
