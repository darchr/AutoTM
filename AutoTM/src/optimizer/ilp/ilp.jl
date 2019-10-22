# General idea here: An ILPOptimizer is a lightweight type used to create the a full model.
# An `ILPHolder` is the struct that holds most of the metadata about ILP construction.
# An `ILPOptimizer` is used to construct an `ILPHolder`.

# ILP Optimizer
abstract type ILPType end

struct Static <: ILPType end
struct Synchronous <: ILPType end
struct Asynchronous <: ILPType end

struct ILPOptimizer{T, U <: ILPType} <: AbstractOptimizer{T} 
    # PMM to DRAM ratio - depending on `T`
    x::T
end

# Hijack `_getratio` to forward to the correct field.
_getratio(I::ILPOptimizer) = I.x

function (O::ILPOptimizer{Rational{Int64}})(data, backend::nGraph.Backend)
    bounds = Profiler.allocation_bounds(data)

    x = fill(round(Int, (bounds.upper_bound / (getratio(O) + 1)) / 1E6), size(nodes(data)))
    println("Trying to use $(maximum(x)) MB of memory")
    return ILPHolder(O, x, backend; defrag = !iszero(_numerator(O)))
end

function (O::ILPOptimizer{Int64})(data, backend::nGraph.Backend)
    x = fill(round(Int, getlimit(O) / 1E6), size(nodes(data)))
    println("Trying to use $(maximum(x)) MB of memory")
    return ILPHolder(O, x, backend)
end

Static(x::T) where {T <: OPTIM_TYPES} = ILPOptimizer{T, Static}(x)
Synchronous(x::T) where {T <: OPTIM_TYPES} = ILPOptimizer{T, Synchronous}(x)
Asynchronous(x::T) where {T <: OPTIM_TYPES} = ILPOptimizer{T, Asynchronous}(x)

#####
##### TensorMeta
#####

struct TensorMeta
    graph::MetaGraph

    # Nodes using this tensor
    users::Vector{XNode}

    # Look-up a node wrapper, get the node that serves as a reference for this
    reference_map::Dict{XNode, XNode}

    # Flag to indicate that this is a fixed tensor - can decrease variable generation.
    isfixed::Bool 
end

#####
##### ILPHolder
#####

mutable struct ILPHolder{T <: ILPType}
    dram_limits::Vector{Int}

    descriptors::Dict{XTensor{XNode}, TensorMeta}
    async_move_vars::Dict{XNode, Vector{JuMP.VariableRef}}

    # Need to key this with string names instead of XNode because of recompiling the ngraph
    # function and reporifling.
    #
    # New XNodes get created that don't hash the same as the old ones.
    node_to_limit_index::Dict{String, Int}

    # Bandwidths
    read_bandwidth::Int64
    write_bandwidth::Int64
    read_bandwidth_async::Int64
    write_bandwidth_async::Int64

    # Flag to determine if we need to defrag
    defrag::Bool
end


# Build an ILP Holder that will constain most of the metadata for IL: Construction.
function ILPHolder(O::ILPOptimizer{<:Any, T}, limits, backend; defrag = true) where {T}
    return ILPHolder{T}(
        limits,
        Dict{XTensor{XNode}, TensorMeta}(),
        Dict{XNode, Vector{JuMP.VariableRef}}(),
        Dict{String, Int}(),
        _bw_remote_local_sync(backend),
        _bw_local_remote_sync(backend),
        _bw_remote_local_async(backend),
        _bw_local_remote_async(backend),
        defrag,
    )
end

name(::Type{Static}) = "static"
name(::Type{Synchronous}) = "synchronous"
name(::Type{Asynchronous}) = "asynchronous"
name(::ILPOptimizer{<:Any, T}) where {T} = name(T)

# Struct to be passed around since all these items are generally used together anyways.
mutable struct Frame{T <: ILPHolder}
    modeltype::T
    model::JuMP.Model
    profile_data::FunctionData

    # Function parameters that are in the "local" memory and thus should be counted for the
    # total memory consumption.
    local_args::Vector{XTensor{XNode}}
end

Frame(modeltype, model::JuMP.Model, profile_data::FunctionData) = 
    Frame(modeltype, model, profile_data, XTensor{XNode}[])

limit(F::Frame) = limit(F.modeltype)
JuMP.optimize!(F::Frame) = optimize!(F.model)

#####
##### Model Types
#####

# Implementations
include("tensor_graphs.jl")
include("formulation.jl")
include("configure.jl")
include("inspect.jl")
include("compare.jl")
