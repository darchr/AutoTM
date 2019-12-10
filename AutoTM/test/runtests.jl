using AutoTM, nGraph
using Test
using Flux

# TODO: Delayed until putting IO into either PMM/DRAM is handled correctly.
#include("profiler.jl")


include("utils.jl")
#include("verify.jl")
include("benchmarker.jl")
