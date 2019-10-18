module Experiments

using Serialization

using ..Utils
using ..Profiler
using ..Optimizer
import ..Zoo
import ..Visualizer
import ..Visualizer.canonical_path
import nGraph

# Make a data directory if needed
function __init__()
    for dir in (DATADIR, CACHEDIR, CPUDATA, GPUDATA)
        !ispath(dir) && mkdir(dir)
    end
end

# Common Paths
const EXPERIMENTS_DIR = @__DIR__
const SRCDIR = dirname(EXPERIMENTS_DIR)
const PKGDIR = dirname(SRCDIR)
const REPODIR = dirname(PKGDIR)
const DATADIR = joinpath(REPODIR, "data")
const CACHEDIR = joinpath(DATADIR, "caches")

const CPUDATA = joinpath(DATADIR, "cpu")
const GPUDATA = joinpath(DATADIR, "gpu")

const SINGLE_KERNEL_PATH = joinpath(DATADIR, "caches", "single_cpu_profile.jls")
const MULTIPLE_KERNEL_PATH = joinpath(DATADIR, "caches", "multiple_cpu_profile.jls")
const GPU_CACHE = joinpath(CACHEDIR, "gpu_profile.jls")

const TILING_FACTOR = Dict(
    SINGLE_KERNEL_PATH => 1,
    MULTIPLE_KERNEL_PATH => 4,
    GPU_CACHE => 1,
)

# Easy model declarations
include("models.jl")
include("conventional.jl")
include("large.jl")
include("gpu.jl")

#####
##### Trials
#####

wrap(x::Union{Tuple, <:Array, <:Iterators.Flatten}) = x
wrap(x) = (x,)

savedir(::nGraph.Backend{nGraph.CPU}) = CPUDATA
savedir(::nGraph.Backend{nGraph.GPU}) = GPUDATA

getcache(::nGraph.Backend{nGraph.CPU}, path::String) = CPUKernelCache(path)
getcache(::nGraph.Backend{nGraph.GPU}, path::String) = GPUKernelCache(path)

# Unwrap the string name from `cache`
canonical_path(f, opt::Optimizer.AbstractOptimizer, cache::String, backend::nGraph.Backend, suffix = nothing) =
    canonical_path(f, Optimizer.name(opt), cache, backend, suffix)

function canonical_path(f, opt::String, cache::String, backend::nGraph.Backend, suffix = nothing)
    # Find the prefix for the cache
    cachename = first(splitext(basename(cache)))
    n = join((name(f), opt, cachename), "_")
    if !isnothing(suffix)
        n = n * "_$suffix"
    end
    n = n * ".jls"
    return joinpath(savedir(backend), n)
end

function execute(fns, opts, caches, backend, suffix = nothing; kw...)
    # Wrap functions, optimizers, and caches so we can safely iterate over everything
    for f in wrap(fns), cache in wrap(caches), opt in wrap(opts)
        savefile = canonical_path(f, opt, cache, backend, suffix)
        Profiler.compare(f, opt, backend;
            statspath = savefile,

            # Set the cache as well as how many times to replicate kernels for profiling
            cache = getcache(backend, cache),
            kw...
        )
    end
end

end
