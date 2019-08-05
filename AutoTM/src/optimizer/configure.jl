# This is how we get the ILP (or other optimization routines) back to ngraph


# Struct for keeping track of what tensors are moved.
#
# children: Maps a tensor `t` to the collection tensors that are the results of "Move"
#   instructions ultimately beginning at `t`.
# parent: Maps a tensor `t` to its parent tensor. The following should hold: `t ∈ children[parent[t]]`
struct TensorMap
    children::Dict{TensorDescriptor, Vector{TensorDescriptor}}
    parent::Dict{TensorDescriptor, TensorDescriptor}
end
TensorMap() = TensorMap(
    Dict{TensorDescriptor, Vector{TensorDescriptor}}(),
    Dict{TensorDescriptor, TensorDescriptor}()
)

function addtensor!(M::TensorMap, d::TensorDescriptor)
    @assert !haskey(M.children, d)
    @assert !haskey(M.parent, d)

    M.children[d] = TensorDescriptor[]
    M.parent[d] = d
end

function addchild!(M::TensorMap, parent::TensorDescriptor, child::TensorDescriptor)
    push!(M.children[parent], child)
    M.parent[child] = parent
end

getchildren(M::TensorMap, parent) = M.children[parent]
getparent(M::TensorMap, child) = M.parent[child]
isparent(M::TensorMap, tensor) = getparent(M, tensor) == tensor

#####
##### Move Action
#####
struct MoveAction
    consumers::Vector{NodeDescriptor}
    location::TensorLocation
    replace_incumbent::Bool
    # Additional data needed for "asynchronous" moves
    concurrent::Union{Nothing, NodeDescriptor}
end
isasync(M::MoveAction) = !isnothing(M.concurrent)

function _initial_loc(path)
    initial_location = first(path).location
    if initial_location == LOC_PMEM
        return PMEM
    elseif isdram(initial_location)
        return DRAM
    else
        error("$(initial_location)???")
    end
end

#####
##### configure!
#####
function configure!(fn::nGraph.NFunction, data::ProfileData, schedule, algos = nothing)
    # Unpack args
    _cleanup!(fn)

    # Get the locations of the tensors currently in the graph
    config = Dict{TensorDescriptor, TensorLocation}()

    # Process the move node chains
    tensor_map = TensorMap()

    # We find all nodes that are targets of a move to DRAM and insert a synchronization
    # barrier.
    #
    # We only do this once for each target because a synchronization barrier will
    # synchronize ALL asynchronous moves
    #
    # KEY: The node that has at least one input coming from an async move
    # VALUE: An async move to this node.
    synced_nodes = Set{nGraph.NodeDescriptor}()

    @showprogress 1 "Computing Move Nodes" for (tensor, (initial_location, actions)) in schedule
        addtensor!(tensor_map, tensor)
        config[tensor] = initial_location

        producer = _producer(tensor, data)
        producer_output = findonly(isequal(tensor), outputs(producer))
        incumbent = tensor

        for action in actions

            consumers = action.consumers
            consumer_inputs = [findonly(isequal(incumbent), inputs(n)) for n in consumers]

            if isasync(action)
                move_node = insert_moveasync_node!(
                    producer,
                    producer_output,
                    consumers,
                    consumer_inputs,
                    action.concurrent;
                )

                if !ismove(producer)
                    @assert data.node_to_index[producer] < data.node_to_index[action.concurrent]
                end
            else
                move_node = insert_move_node!(
                    producer,
                    producer_output,
                    consumers,
                    consumer_inputs,
                )
            end

            # Add the new output tensor tothe tensor map
            for output in outputs(move_node)
                addchild!(tensor_map, tensor, output)
            end

            # Quick debug
            if action.location == PMEM && !isasync(action)
                @assert initial_location == DRAM
            end

            # Add this move node to `node_dict` and assign its output tensor to the config.
            output_tensor = first(outputs(move_node))
            config[output_tensor] = action.location

            if action.replace_incumbent
                producer = move_node
                # Since we're just inserting move nodes, the output index will now always
                # be 1
                producer_output = 1
                incumbent = output_tensor
            end
        end
    end

    #####
    ##### Apply the config
    #####

    # Iterate over each node and each output tensor for each node. Each output tensor should
    # have an assigned location
    for node in fn, output in outputs(NodeDescriptor(node))
        if config[output] == PMEM
            make_persistent(output)
        end
    end

    # Run priority pass after configuration
    priority_pass!(fn)

    # Set algorithms and workspaces
    return tensor_map
end

function configure!(fex::nGraph.FluxExecutable, data::ProfileData, schedule)
    f = fex.ex.ngraph_function
    tensor_map = configure!(f, data, schedule)

    @info "Recompiling Function"
    fex = nGraph.recompile(fex)

    return fex, tensor_map
end

