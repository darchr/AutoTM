module Visualizer

import nGraph
using ..Utils

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

end
