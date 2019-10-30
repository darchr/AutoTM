module Benchmarker

using AutoTM
using nGraph
using PGFPlotsX

using AutoTM.Utils

using Serialization

# Make data directories if needed.
function __init__()
    for dir in (DATADIR, CPUDATA, GPUDATA, FIGDIR)
        isdir(dir) || mkpath(dir)
    end

    # Configure the PGFPlotsX backend
    PGFPlotsX.latexengine!(PGFPlotsX.PDFLATEX)
end

const SRCDIR = @__DIR__
const PKGDIR = dirname(SRCDIR)
const DATADIR = joinpath(PKGDIR, "data")
const FIGDIR = joinpath(PKGDIR, "figures")

const CPUDATA = joinpath(DATADIR, "cpu")
const GPUDATA = joinpath(DATADIR, "gpu")

include("benchmarks.jl")
include("plots.jl")

#####
##### Trials
#####

# Forward the `name` function to `AutoTM.Experiments.name`
name(x...) = AutoTM.Experiments.name(x...)

wrap(x::Union{Tuple, <:Array, <:Iterators.Flatten}) = x
wrap(x) = (x,)

savedir(::Type{nGraph.CPU}) = CPUDATA
savedir(::Type{nGraph.GPU}) = GPUDATA
savedir(::nGraph.Backend{T}) where {T} = savedir(T)

getcache(::nGraph.Backend{nGraph.CPU}, path::String) = AutoTM.Profiler.CPUKernelCache(path)
getcache(::nGraph.Backend{nGraph.GPU}, path::String) = AutoTM.Profiler.GPUKernelCache(path)

# Unwrap the string name from `cache`
function canonical_path(
        f,
        opt::AutoTM.Optimizer.AbstractOptimizer,
        cache::String,
        backend::nGraph.Backend,
        suffix = nothing
    )
    return canonical_path(f, AutoTM.Optimizer.name(opt), cache, backend, suffix)
end

function canonical_path(f, opt::String, cache::String, backend, suffix = nothing)
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
