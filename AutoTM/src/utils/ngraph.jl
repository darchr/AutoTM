"""
$(SIGNATURES)

Insert an nGraph `move` node between `producer` at output `index` and all `consumers`. 
Return the newly created node.

If optional argument `concurrent` is provided, the generated `move` node will be happen 
across `concurrent`.
"""
function insert_move_node!(
        producer::NodeDescriptor, 
        index, 
        consumers::Vector{NodeDescriptor}, 
        consumer_inputs,
        concurrent = nothing
    )

    if isnothing(concurrent)
        move_node = nGraph.move(nGraph.Node(producer), index)
    else
        move_node = nGraph.moveasync(nGraph.Node(producer), index, nGraph.Node(concurrent))
    end
    for (consumer, input) in zip(consumers, consumer_inputs)
        nGraph.splice(nGraph.Node(producer), index, nGraph.Node(consumer), input, move_node)
    end

    return NodeDescriptor(move_node)
end

#####
##### General helpers
#####

const NON_PROFILED_NODES = (
    "Parameter",
    "Constant", 
    "Move", 
    "MoveAsync", 
    "Result",
)

"""
$(SIGNATURES)

Return `true` if nodelike `x` should be profiled.
"""
hasprofile(description::String) = !in(description, NON_PROFILED_NODES)
hasprofile(x::nGraph.NodeLike) = hasprofile(nGraph.description(x))

"""
$(SIGNATURES)

Return `true` if nodelike `x` is a `Move` node.
"""
ismove(description::String) = startswith(description, "Move")
ismove(x::nGraph.NodeLike) = ismove(nGraph.description(x))

"""
$(SIGNATURES)

Return `true` if nodelike `x` is a `MoveAsync` node.
"""
ismoveasync(description::String) = startswith(description, "MoveAsync")
ismoveasync(x::nGraph.NodeLike) = ismoveasync(nGraph.description(x))

"""
$(SIGNATURES)

Return `true` if nodelike `x` is a synchronous `Move` node.
"""
ismovesync(description::String) = description == "Move"
ismovesync(x::nGraph.NodeLike) = ismovesync(nGraph.description(x))

"""
$(SIGNATURES)

Return `true` if nodelike `x` is a constant.
"""
isconstant(description::String) = startswith(description, "Constant")
isconstant(x::nGraph.NodeLike) = isconstant(nGraph.description(x))

"""
$(SIGNATURES)

Return `true` if nodelike `x` is a parameter.
"""
isparam(str::String) = startswith(str, "Parameter")
isparam(t) = isparam(nGraph.description(t))

"""
$(SIGNATURES)

Return `true` is nodelike `x` is a result.
"""
isresult(str::String) = startswith(str, "Result")
isresult(t) = isresult(nGraph.description(t))

"""
$(SIGNATURES)

Return all of input tensor descriptors for `fex`.
"""
input_tensors(fex::nGraph.FluxExecutable) = input_tensors(fex.ex.ngraph_function)
function input_tensors(f::nGraph.NFunction)
    params = NodeDescriptor.(nGraph.get_parameters(f))
    return Iterators.flatten(outputs.(params))
end

"""
$(SIGNATURES)

Return all of the output tensor descriptors for `fex`.
"""
output_tensors(fex::nGraph.FluxExecutable) = output_tensors(fex.ex.ngraph_function)
function output_tensors(f::nGraph.NFunction)
    params = NodeDescriptor.(nGraph.get_results(f))
    return Iterators.flatten(outputs.(params))
end

"""
$(SIGNATURES)

Mark that `tensor` should be allocated into PMM.
"""
make_persistent(tensor::TensorDescriptor) = nGraph.make_persistent(tensor)

#####
##### For managing callbacks
#####

"""
Mutation of the ngraph graph happens via callbacks that are called by ngraph during the 
function compilation process.

At the time of writing this docstring, there are two callback sites in the modified ngraph:
one right before memory assignment and one right after memory assignment.

Both of these callbacks are passed an `nGraph.NFunction`.

The `CallbackChain` is a way of setting up these callbacks easily to seemlessly allow 0, 1, 
or 2 callbacks to be passed to ngraph without trouble.

Functions will be called in the order they are added via [`AutoTM.Utils.callback!`](@ref)

$(METHODLIST)
"""
mutable struct CallbackChain
    # Vector of callback functions
    fns::Vector
    # Number of times this has been invoked.
    num_invocations::Int64
end
CallbackChain() = CallbackChain([], 1)

"""
$(SIGNATURES)

Add `f` to the [`CallbackChain`](@ref) `chain`.
"""
callback!(chain::CallbackChain, f) = push!(chain.fns, f)

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
"""
Custom exception for gracefully exiting out of ngraph callbacks.
"""
struct CompilerExit <: Exception end

