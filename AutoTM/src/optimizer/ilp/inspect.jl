estimate_move_time(fex::nGraph.FluxExecutable, frame::Frame{ILPHolder{Static}}) = zero(Float64)
estimate_move_time(fex::nGraph.FluxExecutable, frame::Frame) = 
    estimate_move_time(fex.ex.ngraph_function, frame)

function estimate_move_time(f::nGraph.NFunction, frame::Frame)
    move_time = zero(Float64)
    for _node in f
        node = NodeDescriptor(_node)

        # If this is a move, determine which direction data is being moved and add the move
        # time estimate to the rolling counter.
        if description(node) == "Move"
            tensor = first(outputs(node))
            if nGraph.is_persistent(tensor)
                move_time += sizeof(tensor) / wb(frame.modeltype)
            else
                move_time += sizeof(tensor) / rb(frame.modeltype)
            end
        end
    end
    return move_time
end


# your moves are weak
function profile_moves(fex)
    timing_data = read_timing_data(fex.ex.ngraph_function)
    computed_stats = Dict{String, NamedTuple}()
    for node_unwrapped in fex.ex.ngraph_function
        node = NodeDescriptor(node_unwrapped)
        ismove(node) || continue

        time = timing_data[findfirst(x -> x["name"] == nGraph.name(node), timing_data)]["dur"]
        # Convert bytes to GB, time from μs to s
        bytes = sizeof(first(inputs(node)))
        bandwidth = (bytes / 1E9) / (time / 1E6)
        computed_stats[nGraph.name(node)] = (
            bytes = bytes,
            bandwidth = bandwidth,
            write_to_pmem = !nGraph.is_persistent(first(inputs(node))),
        )
    end

    # Summarize read and write bandwidth
    println("Read Bandwidths")
    for f in (ismoveasync, !ismoveasync)
        s = 0
        count = 0
        for (node_name, stats) in computed_stats
            f(node_name) || continue
            if stats.write_to_pmem == false
                println("$node_name => $(stats.bandwidth) GB/s")
                println("    size: $(stats.bytes) B")

                if !isinf(stats.bandwidth)
                    s += stats.bandwidth
                    count += 1
                end
            end
        end

        println()
        println("Average Bandwidth: ", s / count)
        println()
    end
    println()
    println("Write Bandwidths")
    for f in (ismoveasync, !ismoveasync)
        s = 0
        count = 0
        for (node_name, stats) in computed_stats
            f(node_name) || continue
            if stats.write_to_pmem == true
                println("$node_name => $(stats.bandwidth) GB/s")
                println("    size: $(stats.bytes) B")

                if !isinf(stats.bandwidth)
                    s += stats.bandwidth
                    count += 1
                end
            end
        end
        println()
        println("Average Bandwidth: ", s / count)
        println()
    end

    return computed_stats
end

# Inspection routines for the post optimized ILP model
function list_overlaps(frame::Frame)
    model = frame.model
    node_times = model[:node_times]
    for node in nodes(frame.profile_data)
        node_name = nGraph.name(node)
        haskey(node_times, node_name) || continue
        _async = get(model[:tensor_async], node_name, nothing)
        if !isnothing(_async) && !iszero(JuMP.value(_async))
            # Get the values of the asynchronous move times
            async_move_time = JuMP.value(_async)
            node_execution_time = JuMP.value(node_times[node_name])

            @info """
            Overlap times for $node_name
            Node Execution Time: $node_execution_time
            Async Move Time : $async_move_time
            """
        end
    end
end
