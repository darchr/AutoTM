module Experiments

using ..Utils
using ..Profiler
using ..Optimizer
import ..Zoo
import nGraph

# Make a data directory if needed
function __init__()
    for dir in (DATADIR, CACHEDIR)
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

# Easy model declarations
include("models.jl")
include("conventional.jl")
include("large.jl")

end
