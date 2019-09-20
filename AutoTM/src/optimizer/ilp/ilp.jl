# Struct to be passed around since all these items are generally used together anyways.
mutable struct Frame{T}
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

# ILP Optimizer
abstract type ILPOptimizer{T} <: AbstractOptimizer{T} end

## Static ILP Formulation
struct Static{T} <: ILPOptimizer{T}
    # PMM to DRAM ratio
    ratio::T

    # We use inner constructors to avoid automitic promotion to rationals or ints which
    # could lead to subtle bugs.
    Static{T}(x::T) where {T <: OPTIM_TYPES} = new{T}(x)
    Static(x::T) where {T <: OPTIM_TYPES} = new{T}(x)
end

name(::Static) = "static"
function (M::Static{Rational{Int64}})(data, backend::nGraph.Backend)
    bounds = Profiler.allocation_bounds(data)

    x = fill(round(Int, (bounds.upper_bound / (getratio(M) + 1)) / 1E6), size(nodes(data)))
    println("Trying to use $(maximum(x)) MB of memory")
    return static(x; defrag = !iszero(_numerator(M)))
end

function (M::Static{Int64})(data, backend::nGraph.Backend)
    x = fill(round(Int, getlimit(M) / 1E6), size(nodes(data)))
    println("Trying to use $(maximum(x)) MB of memory")
    return static(x)
end


## Synchronous ILP Formulation
struct Synchronous{T} <: ILPOptimizer{T}
    ratio::T
    Synchronous{T}(x::T) where {T <: OPTIM_TYPES} = new{T}(x)
    Synchronous(x::T) where {T <: OPTIM_TYPES} = new{T}(x)
end

name(::Synchronous) = "synchronous"
function (M::Synchronous{Rational{Int}})(data, backend::nGraph.Backend)
    bounds = Profiler.allocation_bounds(data)
    x = fill(
        round(Int, (bounds.upper_bound / (getratio(M) + 1)) / 1E6), size(nodes(data))
    )
    println("Trying to use $(maximum(x)) MB of memory")
    return synchronous(x,
                              _bw_remote_local_sync(backend),
                              _bw_local_remote_sync(backend);
                              defrag = !iszero(_numerator(M))
                             )
end

function (M::Synchronous{Int})(data, backend::nGraph.Backend)
    x = fill(round(Int, getlimit(M) / 1E6), size(nodes(data)))
    println("Trying to use $(maximum(x)) MB of memory")
    return synchronous(x,
                              _bw_remote_local_sync(backend),
                              _bw_local_remote_sync(backend);
                              defrag = !iszero(_numerator(M))
                             )
end


## Asynchronous ILP Formulation
struct Asynchronous{T} <: ILPOptimizer{T}
    ratio::T

    Asynchronous{T}(x::T) where {T <: OPTIM_TYPES} = new{T}(x)
    Asynchronous(x::T) where {T <: OPTIM_TYPES} = new{T}(x)
end

name(::Asynchronous) = "asynchronous"
function (M::Asynchronous{Rational{Int}})(data, backend::nGraph.Backend)
    bounds = Profiler.allocation_bounds(data)
    x = fill(round(Int, (bounds.upper_bound / (getratio(M) + 1)) / 1E6), size(nodes(data)))
    println("Trying to use $(maximum(x)) MB of memory")
    return asynchronous(x,
                              _bw_remote_local_sync(backend),
                              _bw_local_remote_sync(backend),
                              _bw_remote_local_async(backend),
                              _bw_local_remote_async(backend);
                              defrag = !iszero(_numerator(M))
                              )
end

function (M::Asynchronous{Int})(data, backend::nGraph.Backend)
    x = fill(round(Int, getlimit(M) / 1E6), size(nodes(data)))
    println("Trying to use $(maximum(x)) MB of memory")
    return asynchronous(x, 
                              _bw_remote_local_sync(backend),
                              _bw_local_remote_sync(backend),
                              _bw_remote_local_async(backend),
                              _bw_local_remote_async(backend);
                              defrag = !iszero(_numerator(M))
                             )
end

#####
##### Model Types
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

# Singleton types for dispatching to different formulations
abstract type ILPFormulationType end
struct IsFixed <: ILPFormulationType end
struct IsSynchronous <: ILPFormulationType end
struct IsAsynchronous <: ILPFormulationType end

mutable struct ILPHolder{T <: ILPFormulationType}
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

# Implementations
include("tensor_graphs.jl")
include("formulation.jl")
include("configure.jl")
include("inspect.jl")
include("compare.jl")
