module Utils

# For compiling
export actualize

# Exports from ioconfig
export IOConfig, setindex, TensorLocation, DRAM, PMEM

# Export from metagraph
export MetaGraph, inedges, outedges, _meta, rem_edge!

# Export fron ngraph
export  insert_move_node!,
        insert_moveasync_node!,
        hasprofile,
        ismemory_intensive,
        ismove,
        ismoveasync,
        ismovesync,
        isconstant,
        isparam,
        isresult,
        _producer,
        _lastuser,
        input_tensors,
        output_tensors,
        make_persistent,
        make_volatile,
        CallbackChain,
        callback!,
        CompilerExit

# Reexport some stuff from nGraph
export TensorDescriptor, NodeDescriptor, inputs, outputs, description

# Ratios
export getratio, ratio_string, footprint

# Random
export find_vertex, find_edge, findonly, dict_push!

import LightGraphs

# Import all of nGraph plus some commonly used names
#
# The commonly used names will be reexported so all dependencies on the local Utils module
# will have those as well
import nGraph
import nGraph: TensorDescriptor, NodeDescriptor, inputs, outputs, description

#####
##### Compile and create a model
#####

function actualize(backend, func; env = (), nkw...)
    f, args, kw = func()
    return withenv(env...) do
        nGraph.compile(backend, f, args...; kw..., nkw...)
    end
end

include("ioconfig.jl")
include("metagraph.jl")
include("ngraph.jl")
include("ratio.jl")

# Random other stuff

"""
    findonly(f, itr)

Find the first element of `x` iterator `itr` where `f(x) == true` and make sure that `x`
is the only element of `itr` with this property.

Return the index of `x`.
"""
function findonly(f, itr)
    idx = findfirst(f, itr)
    isnothing(idx) && error()
    return idx
end

dict_push!(d, k, v) = haskey(d, k) ? push!(d[k], v) : (d[k] = [v])

#####
##### Utility Functions
#####

function find_vertex(g, f)
    iter = filter(v -> f(g,v), collect(LightGraphs.vertices(g)))
    # Make sure we only have one match
    @assert length(iter) == 1
    return first(iter)
end

function find_edge(g, f)
    iter = filter(e -> f(g,e), collect(LightGraphs.edges(g)))

    # Make sure we only have one match
    @assert length(iter) == 1
    return first(iter)
end

end
