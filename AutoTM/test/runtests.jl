using AutoTM, nGraph
using Test
using Flux

# TODO: Delayed until putting IO into either PMM/DRAM is handled correctly.
#include("profiler.jl")

#####
##### Test some of the "utils" functionality.
#####

include("utils.jl")
include("verify.jl")
