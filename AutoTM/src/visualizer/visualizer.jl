module Visualizer

import nGraph
using ..Utils
using ..Optimizer: Optimizer
using ..Profiler: Profiler

using PGFPlotsX
using Serialization

function titlename end
function canonical_path end

include("util.jl")
include("cost.jl")
include("cost_performance.jl")
include("error.jl")
include("front.jl")
include("gpu.jl")
include("large.jl")
include("speedup.jl")
include("stats.jl")

# Plot the heap locations of all intermediate tensors.
include("memory.jl")

end
