module Experiments

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
const TILING_FACTOR = Dict(
    SINGLE_KERNEL_PATH => 1,
    MULTIPLE_KERNEL_PATH => 4,
)

# Easy model declarations
include("models.jl")
include("conventional.jl")
include("large.jl")

wrap(x::Union{Tuple, <:Array, <:Iterators.Flatten}) = x
wrap(x) = (x,)

savedir(::nGraph.Backend{nGraph.CPU}) = CPUDATA
savedir(::nGraph.Backend{nGraph.GPU}) = GPUDATA

getcache(::nGraph.Backend{nGraph.CPU}, path::String) = CPUKernelCache(path)

canonical_path(f, opt::Optimizer.AbstractOptimizer, cache::String, backend::nGraph.Backend) =
    canonical_path(f, Optimizer.name(opt), cache, backend)

function canonical_path(f, opt::String, cache::String, backend::nGraph.Backend)
    # Find the prefix for the cache
    cachename = first(splitext(basename(cache))) 
    return joinpath(
        savedir(backend), 
        join((name(f), opt, cachename), "_") * ".jls"
    )
end

function execute(fns, opts, caches, backend; kw...)
    # Wrap functions, optimizers, and caches so we can safely iterate over everything
    for f in wrap(fns), cache in wrap(caches), opt in wrap(opts)
        savefile = canonical_path(f, opt, cache, backend)
        Profiler.compare(f, opt, backend; 
            statspath = savefile, 

            # Set the cache as well as how many times to replicate kernels for profiling
            cache = getcache(backend, cache),
            kernel_tiling = TILING_FACTOR[cache],
        )
    end
end

end
