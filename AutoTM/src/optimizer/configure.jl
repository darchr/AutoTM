#####
##### Move Action
#####
struct MoveAction
    consumers::Vector{XNode}
    location::TensorLocation
    replace_incumbent::Bool
    # Additional data needed for "asynchronous" moves
    concurrent::Union{Nothing, XNode}
end
isasync(M::MoveAction) = !isnothing(M.concurrent)

# Define this at the XNode level because we have to correctly maintain our "shadow" copy
# of the nGraph graph
function insert_move_node!(
        this_producer::XNode,
        producer_output::Integer,
        consumers::Vector{XNode},
        consumer_inputs::Vector{T},
        action,
    ) where {T <: Integer}

    # First, mutate the nGraph graph underneath all the Julia types
    #
    # Dispatch to the correct implementation depending on whether or not this is an 
    # asynchronous move of just a normal synchronous move.
    if !isasync(action)
        move_node = Utils.insert_move_node!(
            unx(this_producer),
            producer_output,
            unx.(consumers),
            consumer_inputs
        )
    else
        move_node = Utils.insert_moveasync_node!(
            unx(this_producer),
            producer_output,
            unx.(consumers),
            consumer_inputs,
            unx(action.concurrent),
        )

        # If this is not a move node, make sure that the concurrent node is scheduled after
        # the producer node
        if !ismove(unx(this_producer))
            @assert this_producer.index < action.concurrent.index
        end
    end

    # Now, create an appropriate xnode
    #
    # Set its index to 0 temporarily
    xnode = XNode(move_node, 0)

    # Remove all consumer tensors from the producer's output tensor
    producer_xtensor = outputs(this_producer)[producer_output]
    push!(xnode.inputs, producer_xtensor)
    deleteat!(producer_xtensor.users, findall(in(consumers), producer_xtensor.users))

    # Create an output xtensor for the newly created node
    xtensor = XTensor(first(outputs(move_node)))
    push!(xtensor.users, xnode)
    push!(xnode.outputs, xtensor)

    for (input, consumer) in zip(consumer_inputs, consumers)
        consumer.inputs[input] = xtensor  
    end

    return xnode
end

#####
##### inner configure!
#####

function configure!(fn::nGraph.NFunction, schedule, data)
    # Get the locations of the tensors currently in the graph
    #
    # NOTE: Leave this as TensorDescriptor since it will ultimately be used for ngraphj
    # configuration.
    config = Dict{TensorDescriptor, TensorLocation}()

    # We find all nodes that are targets of a move to DRAM and insert a synchronization
    # barrier.
    #
    # We only do this once for each target because a synchronization barrier will
    # synchronize ALL asynchronous moves
    #
    # KEY: The node that has at least one input coming from an async move
    # VALUE: An async move to this node.
    synced_nodes = Set{XNode}()

    for (tensor, (initial_location, actions)) in schedule
        config[unx(tensor)] = initial_location

        this_producer = producer(tensor)
        producer_output = findonly(isequal(tensor), outputs(this_producer))
        incumbent = tensor

        for action in actions
            consumers = action.consumers
            consumer_inputs = [findonly(isequal(incumbent), inputs(n)) for n in consumers]

            move_node = insert_move_node!(
                this_producer,
                producer_output,
                consumers,
                consumer_inputs,
                action
            )

            # Quick debug
            if action.location == PMEM && !isasync(action)
                @assert initial_location == DRAM
            end

            # Add this move node to `node_dict` and assign its output tensor to the config.
            output_tensor = first(outputs(move_node))
            config[unx(output_tensor)] = action.location

            if action.replace_incumbent
                this_producer = move_node
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

    # Make a map from tensor descriptors to XTensors.
    # This will be returned to notify upstream processes which inputs should be in PMM
    arg_descriptors = Set(t.tensor for t in tensors(data) if isarg(t))
    remote_args = Set{TensorDescriptor}()
    
    # Iterate over each node and each output tensor for each node. Each output tensor should
    # have an assigned location
    for node in fn, output in outputs(NodeDescriptor(node))
        if config[output] == PMEM
            make_persistent(output)

            # Add the xtensor to the list of persistent args if applicable
            if in(output, arg_descriptors)
                push!(remote_args, output)
            end
        end
    end

    # Run priority pass after configuration
    priority_pass!(fn)

    # Set algorithms and workspaces
    return remote_args
end

