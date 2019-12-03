using ..Utils.AllocatorModel

#####
##### Numa Optimizer Type
#####
struct Numa{T} <: AbstractOptimizer{T}
    ratio::T

    Numa{T}(x::T) where {T <: OPTIM_TYPES} = new{T}(x)
    Numa(x::T) where {T <: OPTIM_TYPES} = new{T}(x)
end

name(::Numa) = "numa"
function (M::Numa{Rational{Int64}})(data, backend::nGraph.Backend)
    bounds = Profiler.allocation_bounds(data)

    # Special case the all DRAM case - give a little more DRAM for fudge-factor sake
    if M.ratio == 0 // 1
        x = round(Int, 1.1 * bounds.upper_bound)
    else
        x = round(Int, (bounds.upper_bound / (M.ratio + 1)))
    end
    println("Trying to use $(x) MB of memory")

    # Return the result as Bytes without scaling to MB
    return x
end

function (M::Numa{Int64})(data, backend::nGraph.Backend)
    x = getlimit(M)
    println("Trying to use $(x) MB of memory")

    # Return the result as Bytes without scaling to MB
    return x
end

#####
##### NUMA First-touch Implementation
######

_ignore(t::String) = any(startswith.(Ref(t), ("Parameter", "Result", "Constant")))
_ignore(t::XTensor) = _ignore(nGraph.name(t))

mutable struct NumaTensorMeta
    tensor::XTensor
    location::TensorLocation
    offset::Int64
end

function numa(backend::nGraph.Backend, f::nGraph.NFunction, opt::Numa, cache)
    data = profile(f, backend; cache = cache)
    limit = opt(data, backend)
    @show limit
    pool = MemoryAllocator(limit, 4096)

    meta = Dict(
        # Default tensors to starting in PMEM
        t => NumaTensorMeta(t, DRAM, -1) for t in tensors(data)
    )

    for (index, node) in enumerate(nodes(data))
        # Try to allocate the outputs in the pool. If we can allocate, these tensors
        # belong in DRAM.
        #
        # Otherwise, they belong in PMEM. Simple as that.
        for tensor in filter(!_ignore, node.newlist)
            offset = allocate(pool, sizeof(tensor))
            if !isnothing(offset)
                meta[tensor].location = DRAM
                meta[tensor].offset = offset
            else
                meta[tensor].location = PMEM
            end
        end

        for tensor in filter(!_ignore, node.freelist)
            # Free this tensor from DRAM
            if meta[tensor].location == DRAM
                free(pool, meta[tensor].offset)
            end
        end
    end

    # Create a schedule and configure the graph
    schedule = Dict(t.tensor => (t.location, MoveAction[]) for t in values(meta))
    return data, schedule, limit
end

function run_numa(backend, f, opt::Numa; cache = nothing, kw...)
    limit_ref = Ref{Any}()
    function cb(f::nGraph.NFunction)
        data, schedule, limit = numa(backend, f, opt, cache)
        configure!(f, schedule, data)
        limit_ref[] = limit

        return nothing
    end

    callbacks = CallbackChain()
    callback!(callbacks, cb)

    A = f()::Actualizer
    fex = nGraph.compile(
        backend,
        A.f,
        A.args...;
        callback = callbacks,
        emit_timing = true,
        A.kw...
    )

    return fex, limit_ref[]
end

#####
##### _compare!
#####
function Profiler._compare!(
        stats,
        f,
        opt::Numa{<:Rational},
        backend::nGraph.Backend{nGraph.CPU};
        skip_run = false,
        cache = nothing,
        kw...,
    )

    fex, limit = ratiosearch(run_numa, backend, f, opt; cache = cache, kw...)

    # Get the predicted run time and then the actual run time
    nt = Dict(
        :dram_limit => limit,
        :ratio => getratio(opt),
        :actual_runtime => gettime(fex),
        :predicted_runtime => 0,
    )

    push!(stats.runs, nt)
    return nothing
end
