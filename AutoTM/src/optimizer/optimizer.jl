module Optimizer

using ..Utils
using ..Profiler
import nGraph
import Match

using ProgressMeter
using LightGraphs
using JuMP, Gurobi

# Scheduling Heuristic
include("affinity.jl")

include("abstractoptimizer.jl")
include("factory.jl")
include("configure.jl")

# Specific optimizer backends
include("ilp/ilp.jl")
include("numa/numa.jl")
include("optimizer_2lm.jl")

#####
##### For gathering statistics
#####

_move_filter() = x -> ismove(x) && !ismoveasync(x)
_async_filter() = x -> ismoveasync(x)

function _move_filter(dest)
    is_persistent_result = (dest == PMEM) ? true : false

    return x -> ismove(x) && nGraph.is_persistent(first(outputs(x))) == is_persistent_result
end

function _async_filter(dest)
    is_persistent_result = (dest == PMEM) ? true : false

    return x -> _async_filter()(x) && nGraph.is_persistent(first(outputs(x))) == is_persistent_result
end

# Count metrics
_count(f, data; kw...) = _count(f, x -> 1, data; kw...)
function _count(f, g, data; filt = x -> true)
    count = 0
    for node in filter(filt, nodes(data))
        for tensor in f(node)
            count += g(tensor)
        end
    end
    return count
end

end
