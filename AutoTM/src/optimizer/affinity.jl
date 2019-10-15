# Some passes for assigning node affinity.
#
# This applies heuristics to nodes like `Broadcast` and `Result` to schedule them in more
# sensible locations

# A follower node is one that should be scheduled as soon as possible
const FOLLOWERS = [
    "Result",
    "Sum",
    "Add",
    "Subtract",
    "MaxPool",
    "AvgPool",
    #"Concat",
    "Max",
    "Min",
    "Multiply",
    "Relu",
    "Reshape",
    "Dot",
]
function is_follower(node)::Bool
    # If this is a move node into remote memory
    if (ismovesync(node) && nGraph.is_persistent(first(outputs(node)))) || ismoveasync(node)
        return true
    end

    # Default set of follower nodes
    description = nGraph.description(node)
    return any(x -> startswith(description, x), FOLLOWERS)
end

# A leader node should be scheduled as late as possible, but clumped before its trailing
# output
const LEADERS = [
    "Broadcast",
    "ConvertLayout",
    "Slice",
]
function is_leader(node)::Bool
    # If this is a move node prefetching memory, do this as soon as possible.
    if (ismovesync(node) && !nGraph.is_persistent(first(outputs(node))))
        return true
    end

    # Default leader nodes
    description = nGraph.description(node)
    return any(x -> startswith(description, x), LEADERS)
end

# Heuristic to assign priorities to nodes in the graph to yield better schedules
function priority_pass!(f::nGraph.NFunction)
    # First, construct a map from nodes to their index in the function
    node_to_index = Dict(NodeDescriptor(n) => i for (i, n) in enumerate(f))

    # The final node priorities to assign. Save this until the end do avoid accidentally
    # mutating the underlying function too early.
    #
    # We avoid assigning priorities to normal nodes, which will just default internally
    # to a "zero" priority
    priorities = Dict{NodeDescriptor, Int64}()

    # Next, assign priorities
    for node in map(NodeDescriptor, f)
        # Async Moves take priority because if they aren't placed right after their async
        # node - codegen will clump them with the wrong computation kernel.
        if ismoveasync(node)
            priorities[node] = -2

        elseif is_follower(node)
            priorities[node] = -1

        elseif is_leader(node)
            # Find the minimum index of all outputs and control dependencies of the node
            deps = nGraph.get_outputs(node)
            p = maximum(get(node_to_index, n, typemax(Int64)) for n in deps)
            priorities[node] = p
        end
    end

    for (n,p) in priorities
        nGraph.set_priority(n, p)
    end
    return nothing
end

#####
##### Scratchpad
#####

# Optimization for 2LM: assign short lived tensors to a different memory pool so they all
# end up at similar memory addresses
function setup_scratchpad!(data::Profiler.FunctionData; threshold = 100)
    for tensor in tensors(data)
        start = producer(tensor).index
        stop = consumer(tensor).index

        if stop - start >= threshold
            nGraph.set_pool_number(tensor.tensor, 2)
        end
    end
    return nothing
end
