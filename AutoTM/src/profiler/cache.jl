# TODO: Hold on to constant parameters such as padding and stride for Convolutions

# Build a dispatch chain for getting kernel parameters
params(x, node::nGraph.NodeLike) = _params(x, node)
params(x, node::XNode) = _params(x, unx(node))

_params(::nGraph.Backend{nGraph.CPU}, node::nGraph.NodeLike) = CPUKernelParams(node)
_params(::nGraph.Backend{nGraph.GPU}, node::nGraph.NodeLike) = GPUKernelParams(node)

# Cache object for recording seen kernels.
struct CPUKernelParams{IS, OS, IT, OT, NIF}
    # The description of the op
    description::String

    # IO Sizes
    input_sizes::IS
    output_sizes::OS
    input_types::IT
    output_types::OT

    # MKLDNN Formats
    ismkl::Bool
    input_formats::NTuple{NIF, Int64}
end

# For
filter_out_io(c::CPUKernelParams) = (
    c.description,
    c.ismkl,
    c.input_formats,
    c.input_types,
    c.output_types
)

# Get the MKL format string for an op
mkldnn_string(x) = last(split(nGraph.Lib.get_mkldnn_string(x), ":"))

function CPUKernelParams(node::nGraph.NodeLike)
    description = nGraph.description(node)

    # Input processing
    num_inputs = nGraph.get_input_size(node)

    input_sizes = ntuple(x -> nGraph.get_input_shape(node, x), num_inputs)
    input_types = ntuple(x -> nGraph.get_input_element_type(node, x), num_inputs)

    # Only
    ismkl = nGraph.is_mkldnn(node)
    input_formats = ntuple(
        # At the moment, still forwarding to nGraph.Lib.
        x -> nGraph.Lib.get_input_format_int(node.ptr, convert(UInt, x-1)),
        num_inputs
    )

    # output processing
    num_outputs = nGraph.get_output_size(node)
    output_sizes = ntuple(x -> nGraph.get_output_shape(node, x), num_outputs)
    output_types = ntuple(x -> nGraph.get_output_element_type(node, x), num_outputs)

    return CPUKernelParams(
        description,
        input_sizes,
        output_sizes,
        input_types,
        output_types,
        ismkl,
        input_formats,
    )
end

## GPU
struct GPUKernelParams{IS, OS, IT, OT}
    # The description of the op
    description::String

    # IO Sizes
    input_sizes::IS
    output_sizes::OS
    input_types::IT
    output_types::OT
end

function GPUKernelParams(node::nGraph.NodeLike)
    description = nGraph.description(node)

    # Input processing
    num_inputs = nGraph.get_input_size(node)
    input_sizes = ntuple(x -> nGraph.get_input_shape(node, x), num_inputs)
    input_types = ntuple(x -> nGraph.get_input_element_type(node, x), num_inputs)

    # output processing
    num_outputs = nGraph.get_output_size(node)
    output_sizes = ntuple(x -> nGraph.get_output_shape(node, x), num_outputs)
    output_types = ntuple(x -> nGraph.get_output_element_type(node, x), num_outputs)

    return GPUKernelParams(
        description,
        input_sizes,
        output_sizes,
        input_types,
        output_types,
    )
end

#####
##### Caches
#####

abstract type AbstractKernelCache end

Base.getindex(cache::AbstractKernelCache, args...) = getindex(cache.cache, args...)
Base.setindex!(cache::AbstractKernelCache, args...) = setindex!(cache.cache, args...)
Base.haskey(cache::AbstractKernelCache, args...) = haskey(cache.cache, args...)
Base.delete!(cache::AbstractKernelCache, args...) = delete!(cache.cache, args...)

# NOTE: Keeping caches as separate types rather than parametric to be backwards compatible
# with older serialized objects.
struct CPUKernelCache <: AbstractKernelCache
    file::String
    cache::Dict{Tuple{CPUKernelParams, IOConfig}, Float64}
end

struct GPUKernelCache <: AbstractKernelCache
    file::String
    cache::Dict{GPUKernelParams, Union{Float64, AlgorithmPerf}}
end

# Method to determine if we can select the algorithm for this kernel.
can_select_algo(c::GPUKernelCache, p::GPUKernelParams) = _cs(c.cache[p])

CPUKernelCache(file; kw...) = _make_cache(CPUKernelCache, file; kw...)::CPUKernelCache
GPUKernelCache(file; kw...) = _make_cache(GPUKernelCache, file; kw...)::GPUKernelCache

function _make_cache(::Type{T}, file; force_new = false) where {T}
    # If the cache path already exists, just return the existing object.
    # The type assertion for the function will make sure we don't return something weird.
    if (force_new == false) && ispath(file)
        # Deserialize the file.
        cache_db = deserialize(file)

        # Check if we have a key for the current environment context
        # TODO: Don't really need this for the GPU ... 
        ctx = _env_context() 
        haskey(cache_db, ctx) && return cache_db[ctx]::T
    end

    # Otherwise, create the object.
    return T(
        file,
        last(fieldtypes(T))()
    )
end

function save(cache::AbstractKernelCache)
    # Make the directory for this cache if needed.
    dir = dirname(cache.file)
    ispath(dir) || mkdir(dir)

    # Deserialize the existing cache.
    ctx = _env_context()
    if ispath(cache.file) 
        cache_db = deserialize(cache.file)::Dict
        cache_db[ctx] = cache
    else
        cache_db = Dict(ctx => cache)
    end

    serialize(cache.file, cache_db)
end

unsafe_load_cache(file) = deserialize(file)

# Methods for working with and filtering caches.
nt_filter(nt::NamedTuple, cache::AbstractKernelCache) = nt_filter(nt, cache.cache)
function nt_filter(nt::NamedTuple, cache::Dict)
    param_config = collect(keys(cache)) 
    filter!(x -> all(getfield(first(x), k) == v for (k,v) in pairs(nt)), param_config)
    return Dict(k => cache[k] for k in param_config)
end

function _filter(nt::NamedTuple, cache::Dict)
    k = unique(first.(collect(keys(cache))))
    filter!(x -> all(getfield(x, k) == v for (k,v) in pairs(nt)), k)
    return k
end

function choices(cache::AbstractKernelCache, sym::Symbol, nt::NamedTuple = NamedTuple())
    # Iterate through the keys in the cache.
    k = _filter(nt, cache.cache)
    return unique(getfield.(k, sym))
end

