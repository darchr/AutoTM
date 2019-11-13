module Counters

# stdlib
using Dates
using Random
using Serialization
using Sockets

# AutoTM dependencies
using AutoTM
using nGraph
using PCM
using SystemSnoop

# External Packages
using DataStructures
using SIMD
using StructArrays
using PGFPlotsX

# We use a `NamedPipe` to communicate from the process that is actually running the
# benchmarks to the process that is recording the counters.
#
# The NamedPipe has the following commands:
#
# * start -- start sampling.
# * stop -- stop sampling and serialize data to the currently set filepath.
# * shutdown -- exit.
# * filepath <payload> -- Set the serialization path to <payload>
# * measurements -- Indicate that a collection of SystemSnoop measurements are ready
#                   to be deserialized from `TRANSFERPATH` (defined below)
# * sampletime <time> -- Configure the time between samples.
const SRCDIR = @__DIR__
const PKGDIR = dirname(SRCDIR)
const DATADIR = joinpath(PKGDIR, "data")
const TEMPDIR = joinpath(PKGDIR, "temp")

const PIPEPATH = joinpath(TEMPDIR, "pipe")
const TRANSFERPATH = joinpath(TEMPDIR, "transfer.jls")
const FIGDIR = joinpath(PKGDIR, "figures")

function __init__()
    # Create temporary directories if needed.
    for dir in (DATADIR, TEMPDIR, FIGDIR)
        !ispath(dir) && mkdir(dir)
    end
end

# General Utility Functions
include("util.jl")

# Bridge into PCM. Also contains code that allows PCM data structures to use the
# SystemSnoop API
include("pcm.jl")

# Setup a server on `PIPEPATH` that will listen and perform requested measurements.
include("server.jl")

# Code for running AutoTM workloads in 1LM or 2LM.
include("autotm.jl")

# Kernels for testing the DRAM cache system.
include("kernels.jl")
include("kernel_runner.jl")

# Plot resultsjo
include("plots.jl")

# Top level entry point for running all benchmarks
include("measurements.jl")
include("benchmarks.jl")

end # module
