# Struct to be passed around since all these items are generally used together anyways.
mutable struct Frame{T}
    modeltype::T
    model::JuMP.Model
    profile_data::ProfileData
end

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

# Implementations
include("formulation.jl")
include("configure.jl")
include("inspect.jl")
include("compare.jl")
