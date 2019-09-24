"""
    insert_move_node!(producer, index, consumers) -> nGraph.Node

Insert an nGraph `move` node between `producer` and all `consumers`. Return the newly
created node.
"""
function insert_move_node!(
        producer::NodeDescriptor, 
        index, 
        consumers::Vector{NodeDescriptor}, 
        consumer_inputs
    )

    move_node = nGraph.move(nGraph.Node(producer), index)
    for (consumer, input) in zip(consumers, consumer_inputs)
        nGraph.splice(nGraph.Node(producer), index, nGraph.Node(consumer), input, move_node)
    end

    return NodeDescriptor(move_node)
end

function insert_moveasync_node!(
        producer::NodeDescriptor,
        index,
        consumers,
        consumer_inputs,
        concurrent,
    )

    move_node = nGraph.moveasync(nGraph.Node(producer), index, nGraph.Node(concurrent))
    for (consumer, input) in zip(consumers, consumer_inputs)
        nGraph.splice(nGraph.Node(producer), index, nGraph.Node(consumer), input, move_node)
    end

    return NodeDescriptor(move_node)
end

#####
##### General helpers
#####
hasprofile(op_description::String) = !in(op_description, ("Parameter", "Constant", "Move", "MoveAsync"))
hasprofile(op::nGraph.Node) = hasprofile(nGraph.description(op))
hasprofile(x::NodeDescriptor) = hasprofile(nGraph.description(x))

# Hook to exclude some nodes from computation overlap
#is_memory_intensive(op_description::String) = in(op_description, ("MatmulBias",))
is_memory_intensive(op_description::String) = false
is_memory_intensive(op::nGraph.Node) = is_memory_intensive(nGraph.description(op))
is_memory_intensive(op::NodeDescriptor) = is_memory_intensive(nGraph.description(op))

ismove(description::String) = startswith(description, "Move")
ismove(x::nGraph.NodeLike) = ismove(nGraph.description(x))

ismoveasync(description::String) = startswith(description, "MoveAsync")
ismoveasync(x::nGraph.NodeLike) = ismoveasync(nGraph.description(x))

ismovesync(description::String) = description == "Move"
ismovesync(x::nGraph.NodeLike) = ismovesync(nGraph.description(x))

isconstant(description::String) = startswith(description, "Constant")
isconstant(x::nGraph.Node) = isconstant(nGraph.description(x))
isconstant(x::NodeDescriptor) = isconstant(nGraph.description(x))

# TODO: These might not be perfect ...
isparam(str::String) = startswith(str, "Parameter")
isparam(t) = isparam(nGraph.description(t))

isresult(str::String) = startswith(str, "Result")
isresult(t) = isresult(nGraph.description(t))

input_tensors(fex::nGraph.FluxExecutable) = input_tensors(fex.ex.ngraph_function)
function input_tensors(f::nGraph.NFunction)
    params = NodeDescriptor.(nGraph.get_parameters(f))
    return Iterators.flatten(outputs.(params))
end

output_tensors(fex::nGraph.FluxExecutable) = output_tensors(fex.ex.ngraph_function)
function output_tensors(f::nGraph.NFunction)
    params = NodeDescriptor.(nGraph.get_results(f))
    return Iterators.flatten(inputs.(params))
end

make_persistent(tensor::TensorDescriptor) = nGraph.make_persistent(tensor)
make_volatile(tensor::TensorDescriptor) = nGraph.make_volatile(tensor)

#####
##### For managing callbacks
#####
mutable struct CallbackChain
    # Vector of callback functions
    fns::Vector
    # Number of times this has been invoked.
    num_invocations::Int64
end
CallbackChain() = CallbackChain([], 1)
callback!(G::CallbackChain, f) = push!(G.fns, f)

function (G::CallbackChain)(args...)
    if G.num_invocations > length(G.fns)
        return nothing
    else
        f = G.fns[G.num_invocations]
        G.num_invocations += 1
        return f(args...)
    end
end

# In case we need to gracefully exit from a GPU compilation callback function
struct CompilerExit <: Exception end

