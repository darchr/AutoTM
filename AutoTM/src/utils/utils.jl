module Utils

# For compiling
export Actualizer, actualize, @closure

# Exports from ioconfig
export IOConfig, setindex, TensorLocation, DRAM, PMEM

# Export from metagraph
export MetaGraph, inedges, outedges, getmeta, rem_edge!

# Export fron ngraph
export  insert_move_node!,
        hasprofile,
        ismemory_intensive,
        ismove,
        ismoveasync,
        ismovesync,
        isconstant,
        isparam,
        isresult,
        input_tensors,
        output_tensors,
        make_persistent,
        CallbackChain,
        callback!,
        CompilerExit

# Reexport some stuff from nGraph
export TensorDescriptor, NodeDescriptor, inputs, outputs, description

# Ratios
export getratio, ratio_string, footprint, compare_ratio

# Random
export approx_one, find_vertex, find_edge, findonly, dict_push!, vflatten

import LightGraphs
import JuMP

# Import all of nGraph plus some commonly used names
#
# The commonly used names will be reexported so all dependencies on the local Utils module
# will have those as well
import nGraph
import nGraph: TensorDescriptor, NodeDescriptor, inputs, outputs, description

using DocStringExtensions

include("ioconfig.jl")
include("metagraph.jl")
include("ngraph.jl")
include("ratio.jl")

# Memory Allocator Model
include("allocator.jl")

#####
##### Compile and create a model
#####

"""
Return type for `AutoTM` compatible functions.

$(FIELDS)
"""
struct Actualizer
    "Function to convert to ngraph."
    f

    "Arguements to pass to `f`."
    args::Tuple

    "Keyword arguments to pass to `nGraph.compile`"
    kw::NamedTuple
    Actualizer(f, x...; kw...) = new(f, x, NamedTuple{keys(kw)}(values(kw)))
end

"""
$(SIGNATURES)

Convert `f` to an ngraph executable running on `backend`. Argument `f` must be callable
with no arguments and return an [`Actualizer`](@ref).

Keyword Arguments
-----------------
* `env`: Tuple of environmental variable pairs to be forwared to `withenv` when compiling `f`.

* `nkw`: Extra keyword or keyword overrides to pass to `nGraph.compile`.
"""
actualize(backend, f; env = (), nkw...) = actualize(backend, f(); env = env, nkw...)
function actualize(backend, A::Actualizer; env = (), nkw...)
    return withenv(env...) do
        nGraph.compile(backend, A.f, A.args...; A.kw..., nkw...)
    end
end

"""
$(SIGNATURES)

A macro that wraps an expression in a closure for deferred calling.

julia> y = 2
2

julia> f = AutoTM.@closure y + 2;

julia> f()
4
"""
macro closure(expr)
    return :(() -> $(esc(expr)))
end

"""
$(SIGNATURES)

Return `true` if `x` is approximately equal to 1. Works on standard variables and
`JuMP.VariableRef`.
"""
approx_one(x; atol = 1e-3) = isapprox(x, one(x); atol = atol)
approx_one(x::JuMP.VariableRef; kw...) = approx_one(JuMP.value(x); kw...)

"""
$(SIGNATURES)

Find the first element of `x` iterator `itr` where `f(x) == true` and make sure that `x`
is the only element of `itr` with this property.

Return the index of `x`.
"""
function findonly(f, itr)
    indices = findall(f, itr)
    if length(indices) != 1
        throw(ArgumentError("Expected to find just one valid index. Instead, found $(length(indices))"))
    end
    return first(indices)
end

"""
$(SIGNATURES)

Check if Dict `d` has key `k`. If so, push `v` to `d[k]`. Otherwise, initialize `d[k]`
to a `[v]`.
"""
dict_push!(d, k, v) = haskey(d, k) ? push!(d[k], v) : (d[k] = [v])

"""
$(SIGNATURES)

Variadic version of `Iterators.flatten`.
"""
vflatten(x...) = Iterators.flatten(x)

end

