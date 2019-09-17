### Tools for dealing with multiple algorithms
struct AlgorithmPerf
    # The enum value used to select this algorithm.
    #
    # This is a detail of cuDNN
    enum::UInt32

    # The execution time of the algorithm - type match with the return value from the C++
    # code.
    time::Float32

    # The number of bytes required for working space
    bytes::UInt64
end

enums(a::Vector{AlgorithmPerf}) = map(x -> x.enum, a)
times(a::Vector{AlgorithmPerf}) = map(x -> x.time, a)
function timeat(a::Vector{AlgorithmPerf}, e::Integer)
    ind = something(findfirst(x -> x.enum == e, a))
    return a[ind].time
end

function bytesat(a::Vector{AlgorithmPerf}, e::Integer)
    ind = something(findfirst(x -> x.enum == e, a))
    return a[ind].bytes
end

# We're going to abstract the notion of a Tensor so we can add a bunch of metadata to it
@enum TensorRole Arg Constant Intermediate

# Make these mutable structs since we do mutate some of the fields and want to maintain
# consistency in dictionaries.
struct XTensor{T}
    tensor::TensorDescriptor
    users::Vector{T}
    role::TensorRole
    size::Int
    locations::Vector{TensorLocation}
end

function XTensor(tensor::TensorDescriptor)
    # Be default, everything can live in DRAM
    locations = [DRAM]

    # Add in tensors that may also live in PMEM
    xtensor = XTensor(tensor, XNode[], role(tensor), sizeof(tensor), locations)
    if !isconstant(xtensor)
        push!(xtensor.locations, PMEM)
    end
    return xtensor
end

# Infer the role of this tensor from its name
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

Utils.isconstant(t::XTensor) = t.role == Constant
isarg(t::XTensor) = t.role == Arg
is_persistent(t::XTensor) = nGraph.is_persistent(unx(t))
adduser!(t::XTensor, n) = !in(n, users(t)) && push!(t.users, n)

users(t::XTensor) = t.users
producer(t::XTensor) = first(t.users)
consumer(t::XTensor) = last(t.users)
Base.sizeof(t::XTensor) = t.size

unx(t::XTensor) = t.tensor
locations(t::XTensor) = t.locations
nGraph.name(t::XTensor) = nGraph.name(t.tensor)
Base.show(io::IO, x::XTensor) = println(io, "XTensor: ", nGraph.name(x))

#####
##### XNode
#####

struct XNode
    node::NodeDescriptor
    index::Int
    timings::Dict{IOConfig, Union{Float64, Vector{AlgorithmPerf}}}
    outputs::Vector{XTensor}
    inputs::Vector{XTensor}
    newlist::Vector{XTensor}
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
        Dict{IOConfig, Union{Float64, Vector{AlgorithmPerf}}}(),
        XTensor[],
        XTensor[],
        XTensor[],
        XTensor[]
    )
end

_length_one(x) = (@assert(isone(length(x))); x)

can_select_algo(n::XNode, c::IOConfig) = isa(n.timings[c], Vector{AlgorithmPerf})
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

#####
##### Profile Data
#####

struct FunctionData{T}
    tensors::Set{XTensor}
    nodes::Vector{XNode}
end

nodes(f::FunctionData) = f.nodes
tensors(f::FunctionData) = f.tensors

function FunctionData(fn::nGraph.NFunction, ::Type{T}) where {T}
    tensors = Set{XTensor{XNode}}()
    nodes = XNode[]

    for (index, unwrapped_op) in enumerate(fn)
        op = NodeDescriptor(unwrapped_op)
        xnode = XNode(op, index)
        push!(nodes, xnode)

        for tensor in outputs(op)
            # Create an XTensor from this output.
            xtensor = XTensor(tensor)
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

function possible_configs(data::FunctionData{T}) where {T}
    configs = Set{Tuple{XNode, IOConfig}}()
    for node in nodes(data)
        hasprofile(node) || continue

        # Get possible configs based on backend type
        pc = possible_configs(node, T)
        for config in pc
            push!(configs, (node, config))
        end
    end
    return configs
end

function possible_configs(node::XNode, ::Type{T}) where {T}
    config_inputs, config_outputs = _configsfor(node, T)
    return map(x -> IOConfig(x...),
        Iterators.product(
            Iterators.product(config_inputs...),
            Iterators.product(config_outputs...),
        )
    )
end

# In the CPU case, we can read directly from either local (DRAM) or remote (PMEM) memory.
# Thus, the only constraint on configurations is the constraint on where tensors can live.
_configsfor(node, ::Type{nGraph.CPU}) =
    [locations(t) for t in inputs(node)], [locations(t) for t in outputs(node)]

# In the GPU case - we can only read/write in local (DRAM) memory.
# Wrap the inner DRAM in a tuple to it gets handled correctly in `Iterators.product`.
_configsfor(node, ::Type{nGraph.GPU}) =
    ([(DRAM,) for _ in inputs(node)], [(DRAM,) for _ in outputs(node)])

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
end

# Set everything back to volatile
_cleanup!(f::nGraph.NFunction) = map(_cleanup!, f)

function _cleanup!(node::nGraph.Node)
    for descriptor in outputs(node)
        nGraph.make_volatile(descriptor)
        nGraph.reset_offset(descriptor)
    end
    for descriptor in inputs(node)
        nGraph.make_volatile(descriptor)
        nGraph.reset_offset(descriptor)
    end
end

