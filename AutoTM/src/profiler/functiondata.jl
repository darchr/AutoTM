### Tools for dealing with multiple algorithms
"""
Algorithm information for CUDNN algorithms that offer implementation selections.

$(FIELDS)

Methods: [`enums`](@ref), [`times`](@ref), [`timeat`](@ref), [`bytesat`](@ref)
"""
struct CUDNNAlgorithm
    """
    The enum value used to select this algorithm.

    This is a detail of cuDNN
    """
    enum::UInt32

    """
    The execution time of the algorithm - type match with the return value from the C++
    code.
    """
    time::Float32

    "The number of bytes required for working space"
    bytes::UInt64
end

"""
$(SIGNATURES)

Return the CUDNN algorithm enums from `algs`.
"""
enums(algs::Vector{CUDNNAlgorithm}) = map(x -> x.enum, algs)

"""
$(SIGNATURES)

Return the CUDNN algorithm times from `algs`.
"""
times(algs::Vector{CUDNNAlgorithm}) = map(x -> x.time, algs)

"""
$(SIGNATURES)

Return the time from `algs` for the algorithm witn enum encoding `enum`.
"""
function timeat(algs::Vector{CUDNNAlgorithm}, enum::Integer)
    ind = something(findfirst(x -> x.enum == enum, algs))
    return algs[ind].time
end

"""
$(SIGNATURES)

Return the working space requirement in bytes  from `algs` for the algorithm witn enum
encoding `enum`.
"""
function bytesat(algs::Vector{CUDNNAlgorithm}, e::Integer)
    ind = something(findfirst(x -> x.enum == e, algs))
    return algs[ind].bytes
end

#####
##### XTensor
#####

# We're going to abstract the notion of a Tensor so we can add a bunch of metadata to it
@enum TensorRole Arg Constant Intermediate

const TENSOR_GROUP_NUM = Ref(0)

"""
Wrapper around `TensorDescriptors` that includes extra metadata useful for generating
the ILP.

$(FIELDS)
"""
mutable struct XTensor{T}
    # Can possible map a single XTensor to multiple tensor descriptors if two args are
    # merged.
    "The `TensorDescriptor` that this `XTensor` represents."
    tensor::TensorDescriptor

    """
    Ordered collection of `XNode`s that reference this XTensor. The first entry is the
    producer of this tensor. All the rest read this tensor.
    """
    users::Vector{T}

    """
    The role of the tensor.
    """
    role::TensorRole

    """
    Flag for it this tensor is fixed at some specific location or not. Helpful for reducing
    the number of variables generated for the ILP.

    If the tensor is not fixed, this should be set to `nothing`.
    """
    fixed_at::Union{Nothing,TensorLocation}

    "Size of the tensor in bytes."
    size::Int

    "Collection of `TensorLocations` that this tensor can occupy."
    locations::Vector{TensorLocation}

    # Metadata for parameters and results
    "Indicate that all tensors in this group are actually the same tensor."
    inplace::Bool

    """
    Indicate the group number of this tensor. All tensors in the same group need to have the
    same PMM/DRAM assignment.

    For now, this should only be tweaked for IO tensors.
    """
    group::Int
end

isfixed(x::XTensor) = !isnothing(x.fixed_at)
fixed_location(x::XTensor) = something(x.fixed_at)

function XTensor(tensor::TensorDescriptor, ::Type{T}; fixed_at = nothing) where {T}
    # Add in tensors that may also live in PMEM
    xtensor = XTensor(
        tensor,
        XNode[],
        role(tensor),
        fixed_at,
        sizeof(tensor),
        TensorLocation[],
        false,
        TENSOR_GROUP_NUM[],
    )

    # Update the group number.
    TENSOR_GROUP_NUM[] += 1

    # Check if this tensor is fixed at a location. If so - just append that location
    if isfixed(xtensor)
        push!(xtensor.locations, fixed_at)
    else
        push!(xtensor.locations, DRAM)
        if !isconstant(xtensor)
            push!(xtensor.locations, PMEM)
        end
    end
    return xtensor
end

isinplace(t::XTensor) = t.inplace
makeinplace(t::XTensor) = (t.inplace = true)

"""
$(SIGNATURES)

Merge `XTensor`s `a` and `b` into the same tensor group, which will be the smallest group
of either `a` or `b`.

CAUTION: Currently doesn't merge all tensors in `a`'s or `b`'s group.
"""
function Base.merge!(a::XTensor, b::XTensor)
    # Sanitize inputs.
    if size(unx(a)) != size(unx(b))
        throw(DimensionMismatch("""
            Trying to merge two XTensors with different tensor sizes: $(size(a)), $(size(b))
            """)
        )
    end
    if eltype(unx(a)) != eltype(unx(b))
        throw(ArgumentError("""
            Trying to merge two XTensors with different eltypes: $(eltype(a)), $(eltype(b))
            """)
        )
    end

    # Set the group of the tensors to whatever the minimum is.
    # This makes merging two groups somewhat difficult since we don't have a way to collect
    # all tensors that belong to a group - but I don't think that will generally be a use
    # case.
    #
    # If that does become a ues-case, we'll have to keep a global tracker to all the live
    # allocated XTensors and use finalizers to clean up the list as XTensors are GC'd.
    n = min(a.group, b.group)
    a.group = n
    b.group = n
end

"""
$(SIGNATURES)

Merge all `XTensor`s in `A` into the same group.
"""
function Base.merge!(A::Vector{<:XTensor})
    # Find the lowest group number and merge everything with that.
    _, indmin = findmin(map(x -> x.group, A))
    map(x -> merge!(x, A[indmin]), A)
    return nothing
end

"""
$(SIGNATURES)

Return the `TensorRole` of a `TensorDescriptor` from it's name.
"""
function role(tensor::TensorDescriptor)
    tensorname = nGraph.name(tensor)
    if isparam(tensorname) || isresult(tensorname)
        role = Arg
    elseif isconstant(tensorname)
        role = Constant
    else
        role = Intermediate
    end
    return role
end

Utils.isconstant(t::XTensor) = (t.role == Constant)
isarg(t::XTensor) = t.role == Arg
is_persistent(t::XTensor) = nGraph.is_persistent(unx(t))
adduser!(t::XTensor, n) = !in(n, users(t)) && push!(t.users, n)

"""
$(SIGNATURES)

Return all users of `xtensor`.
"""
users(xtensor::XTensor) = xtensor.users

"""
$(SIGNATURES)

Return the XNode the produces `xtensor`.
"""
producer(xtensor::XTensor) = first(xtensor.users)

"""
$(SIGNATURES)

Return the last `XNode` that uses `xtensor`.
"""
consumer(xtensor::XTensor) = last(xtensor.users)

"""
$(SIGNATURES)

Return the allocation size of `xtensor`.
"""
Base.sizeof(xtensor::XTensor) = xtensor.size

"""
$(SIGNATURES)

Get the underlying `nGraph.TensorDescriptor` from `xtensor`.
"""
unx(xtensor::XTensor) = xtensor.tensor
locations(t::XTensor) = t.locations
nGraph.name(t::XTensor) = nGraph.name(t.tensor)
Base.show(io::IO, x::XTensor) = println(io, "XTensor: ", nGraph.name(x))
getoffset(t::XTensor) = nGraph.get_pool_offset(t.tensor)

#####
##### XNode
#####

"""
Wrapper for `nGraph.NodeDescriptor` with extra metadata.

$(FIELDS)
"""
mutable struct XNode
    "The `NodeDescriptor` this `XNode` represents"
    node::NodeDescriptor

    "The execution number of this node in the topological sort of the graph"
    index::Int

    """
    Timings for each possible `IOConfig` of the node. The collection of `IOConfigs` is
    decided at runtime based off the possible locations of the input and output tensors
    of this node.
    """
    timings::Dict{IOConfig, Union{Float64, Vector{CUDNNAlgorithm}}}

    "Ordered collection of output tensors for this node."
    outputs::Vector{XTensor}

    "Ordered collection of input tensors for this node."
    inputs::Vector{XTensor}

    "Tensors who begin life at this node."
    newlist::Vector{XTensor}

    "Tensors that may be freed after the execution of this node."
    freelist::Vector{XTensor}
end

function XNode(node::NodeDescriptor, index)
    # Don't determine the inputs or outputs yet.
    #
    # Need to make sure all XTensors and XNodes are consistent.
    # We leave this to the FunctionData constructor.
    return XNode(
        node,
        index,
        Dict{IOConfig, Union{Float64, Vector{CUDNNAlgorithm}}}(),
        XTensor[],
        XTensor[],
        XTensor[],
        XTensor[],
    )
end

_length_one(x) = (@assert(isone(length(x))); x)

can_select_algo(n::XNode, c::IOConfig) = isa(n.timings[c], Vector{CUDNNAlgorithm})
can_select_algo(n::XNode) = any(c -> can_select_algo(n, c), configs_for(n))

settime!(n::XNode, c::IOConfig, time) = n.timings[c] = time
configs_for(n::XNode) = keys(n.timings)
config_for(n::XNode) = configs_for(n) |> collect |> _length_one |> first

gettime(n::XNode, c::IOConfig) = n.timings[c]
gettime(n::XNode) = gettime(n, config_for(n))
unx(n::XNode) = n.node

Utils.hasprofile(n::XNode) = hasprofile(n.node)

nGraph.outputs(n::XNode) = n.outputs
nGraph.inputs(n::XNode) = n.inputs
nGraph.name(n::XNode) = nGraph.name(n.node)
Base.show(io::IO, n::XNode) = println(io, "XNode: ", nGraph.name(n))

Base.isless(a::XNode, b::XNode) = isless(a.index, b.index)

#####
##### Profile Data
#####

struct FunctionData{T}
    tensors::Set{XTensor}
    nodes::Vector{XNode}
end

nodes(f::FunctionData) = f.nodes
tensors(f::FunctionData) = f.tensors

FunctionData(fn::nGraph.NFunction, ::nGraph.Backend{T}) where {T} = FunctionData(fn, T)
function FunctionData(fn::nGraph.NFunction, ::Type{T}) where {T}
    tensors = Set{XTensor{XNode}}()
    nodes = XNode[]

    for (index, unwrapped_op) in enumerate(fn)
        op = NodeDescriptor(unwrapped_op)
        xnode = XNode(op, index)
        push!(nodes, xnode)

        for tensor in outputs(op)
            if isresult(op) || isparam(op)
                xtensor = XTensor(tensor, T; fixed_at = DRAM)
            else
                xtensor = XTensor(tensor, T)
            end
            # Create an XTensor from this output.
            #xtensor = XTensor(tensor, T)
            push!(tensors, xtensor)

            # Register the producing node as the first user
            adduser!(xtensor, xnode)

            # Register the xtensor as an output of the xnode
            push!(xnode.outputs, xtensor)
        end

        # Do the same thing for the node inputs
        for tensor in inputs(op)
            # Get the xtensor we made previously from the set of all xtensors
            xtensor = _get(tensors, tensor)

            # Register users and inputs
            adduser!(xtensor, xnode)
            push!(xnode.inputs, xtensor)
        end
    end

    # Run liveness analysis on the set of nodes.
    liveness!(nodes, tensors)

    return FunctionData{T}(tensors, nodes)
end

function _get(a::Set{XTensor{XNode}}, t::TensorDescriptor)
    for i in a
        i.tensor == t && return i
    end
    throw(KeyError(t))
end

#####
##### Liveness Analysis
#####

function liveness!(nodes::Vector{XNode}, tensors::Set{XTensor{XNode}})
    # forward pass
    for op in nodes
        empty!(op.newlist)
        append!(op.newlist, filter(x -> !isarg(x) && !isconstant(x), outputs(op)))
    end

    # add all of the argument tensors to the liveness list for the first op
    ind = findfirst(hasprofile, nodes)
    append!(nodes[ind].newlist, filter(x -> isarg(x), tensors))

    # backward pass
    freed_tensors = Set{XTensor{XNode}}()
    for op in reverse(nodes)
        empty!(op.freelist)
        for tensor in inputs(op)
            if !in(tensor, freed_tensors) && !isarg(tensor) && !isconstant(tensor)
                push!(op.freelist, tensor)
                push!(freed_tensors, tensor)
            end
        end
    end
    ind = findlast(hasprofile, nodes)
    append!(nodes[ind].freelist, filter(x -> isarg(x), tensors))

    # Now, we need to do a cleanup phase.
    #
    # The BatchNorm pass can mess up some aspects of outputs from liveness analysis.
    # Here, we check to see if any tensor shows up in the new list but not the free list.
    # If so, we set its free point to the place where it was created.
    #
    # I hate batchnorm
    #
    # TODO: Fix this because it's harming liveness analysis for arguments.
    tensor_start = Dict{XTensor, Int}()
    for (index, op) in enumerate(nodes), tensor in op.newlist
        tensor_start[tensor] = index
    end

    for op in nodes, tensor in op.freelist
        @assert haskey(tensor_start, tensor)
        delete!(tensor_start, tensor)
    end

    for (tensor, index) in tensor_start
        isarg(tensor) || push!(nodes[index].freelist, tensor)
    end

    return nothing
end

# Convenience for iterating over live tensors
struct LiveTensorIterator
    data::FunctionData
    live_tensors::Set{XTensor{XNode}}
end

live_tensors(data::FunctionData) = LiveTensorIterator(data, Set{XTensor{XNode}}())
Base.length(L::LiveTensorIterator) = length(L.data.nodes)
function Base.iterate(L::LiveTensorIterator, s = 1)
    s > length(L) && return nothing

    # Free tensors from the previous iteration
    if !isone(s)
        for tensor in L.data.nodes[s-1].freelist
            delete!(L.live_tensors, tensor)
        end
    end

    # Add new tensors for this iteration
    for tensor in L.data.nodes[s].newlist
        push!(L.live_tensors, tensor)
    end

    return L.live_tensors, s+1
end


"""
    allocation_bounds(data::FunctionData)

Return upper and lower bounds on the amount of DRAM required for input, output,
constant, and intermediate tensors.

Upper bound is determined by the maximum tensors concurrently live.

Lower bound is determined by the total size of input, output, and constant tensors.
"""
function allocation_bounds(data::FunctionData)
    lower_bound = 0

    # Compute Upper Bound
    upper_bound = 0
    for tensors in live_tensors(data)
        if !isempty(tensors)
            upper_bound = max(upper_bound, sum(sizeof(n) for n in tensors))
        end
    end

    return (upper_bound = upper_bound, lower_bound = lower_bound)
end

#####
##### Valid locations that a tensor can live
#####

function possible_configs(backend::nGraph.Backend, data::FunctionData)
    configs = Set{Tuple{XNode, IOConfig}}()
    for node in nodes(data)
        hasprofile(node) || continue

        # Get possible configs based on backend type
        pc = possible_configs(backend, node)
        for config in pc
            push!(configs, (node, config))
        end
    end
    return configs
end

function possible_configs(::nGraph.Backend{nGraph.CPU}, node::XNode)
    config_inputs = [locations(t) for t in inputs(node)]
    config_outputs = [locations(t) for t in outputs(node)]

    # First - generate all of the configs generated by the Cartesian product of input and
    # output locations.
    configs = vec(map(x -> IOConfig(x...),
        Iterators.product(
            Iterators.product(config_inputs...),
            Iterators.product(config_outputs...),
        )
    ))

    # Filter out configs where some tensors belong to the same group and hence whose
    # locations should vary together>
    tensors = vcat(inputs(node), outputs(node))
    unique_groups = unique(map(x -> x.group, tensors))
    group_indices = [findall(x -> x.group == g, tensors) for g in unique_groups]
    for indices in group_indices
        filter!(x -> isone(length(unique(x[indices]))), configs)
    end

    return configs
end

function possible_configs(::nGraph.Backend{nGraph.GPU}, node::XNode)
    config = IOConfig(
        ntuple(_ -> DRAM, length(inputs(node))),
        ntuple(_ -> DRAM, length(outputs(node))),
    )
    return (config,)
end

function getconfig(n::nGraph.Node)
    f = x -> nGraph.is_persistent(x) ? PMEM : DRAM
    input = map(f, inputs(n)) |> Tuple
    output = map(f, outputs(n)) |> Tuple

    return IOConfig(input, output)
end

#####
##### Setup and cleanup code
#####

function _setup!(node::nGraph.Node, config::IOConfig)
    # Outputs
    for (i, location) in enumerate(config.outputs)
        if location == PMEM
            nGraph.make_persistent(nGraph.output(node, i))
        end
    end

    # Inputs
    for (i, location) in enumerate(config.inputs)
        if location == PMEM
            nGraph.make_persistent(nGraph.input(node, i))
        end
    end
    return nothing
end

