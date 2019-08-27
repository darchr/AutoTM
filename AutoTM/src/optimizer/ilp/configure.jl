# Overload for a Frame
function configure!(f::nGraph.NFunction, frame::Frame)
    # Get initial schedules for the frame
    initial_schedule = get_schedule(frame)

    # Convert this into an appropriate format for the inner `configure!`
    schedule = Dict(
        t => (_initial_loc(path), getactions(tensor_graph, path))
        for (t, (tensor_graph, path)) in initial_schedule
    )

    # TODO: Move this into the innermost `configure!`
    data = frame.profile_data
    for node in nodes(data)
        if nGraph.Lib.can_select_algo(nGraph.getpointer(unx(node)))
            # Only access this once we know that there is at least one node where the
            # algorithm can be decided.
            #
            # Otherwise, `frame.model[:algo_var]` will not be defined.
            algo_var = frame.model[:algo_var]
            count = 0
            local algo_enum
            for enum in enums(gettime(node))
                if approx_one(algo_var[node, enum])
                    count += 1
                    algo_enum = enum
                end
            end

            # Only one algorithm should be selected
            @assert count == 1
            nGraph.Lib.set_algo(
                nGraph.getpointer(unx(node)),
                convert(UInt, algo_enum),
                convert(UInt, Profiler.bytesat(gettime(node), algo_enum))
            )
        end
    end

    @info "Calling Inner Configure"
    return configure!(f, schedule)
end

# Get the path of the tensor traced through the graph
function get_schedule(F::Frame)
    data = F.profile_data
    model_graphs = F.model[:tensor_graphs]

    schedule = Dict{XTensor{XNode}, Tuple{MetaGraph, Vector{VertexMetadata}}}()

    for tensor in tensors(data)
        g = graph(descriptor(F, tensor))

        # Trace the route taken through the graph
        v = find_vertex(g, (g, v) -> _meta(g, v).location == LOC_SOURCE)

        path = [_meta(g, v)]
        seen = Int[]
        while _meta(g, v).location != LOC_SINK
            if isempty(outedges(g, v)) || in(v, seen)
                error("""
                $tensor
                $(_meta(g, v))
                """)
            end

            push!(seen, v)
            for e in outedges(g, v)
                if approx_one(model_graphs[tensor, e])
                    v = dst(e)
                    break
                end
            end
            push!(path, _meta(g, v))
        end
        # Drop the first source element and last sink element
        popfirst!(path)
        pop!(path)

        schedule[tensor] = (g, path)
    end

    return schedule
end

# Consume all of the PKEEP nodes.
function getkeeps(vertices::Vector{VertexMetadata}, index)
    keeps = XNode[]
    while checkbounds(Bool, vertices, index) && isdram(vertices[index].location)
        vertex = vertices[index]
        if vertex.isuser
            push!(keeps, vertices[index].op)
        end
        index += 1
    end
    return unique(keeps)
end

function isasync(tensor_graph, a::VertexMetadata, b::VertexMetadata)
    # Get the vertex number from the metadata - construct the edge
    src = a.vertex_number
    dst = b.vertex_number
    edge_metadata = _meta(tensor_graph, edgetype(tensor_graph)(src, dst))
    return isasync(edge_metadata)
end

# Return `true` if there is an implied write to
write_to_pmem(a, b) = a == LOC_DRAM_PRE && ispmem(b)
read_from_pmem(a, b) = ispmem(a) && isdram(b)

function getactions(tensor_graph, vertices::Vector{VertexMetadata})
    actions = MoveAction[]
    written_to_pmem = false

    for i in Iterators.drop(eachindex(vertices), 1)
        src = vertices[i-1]
        dst = vertices[i]
        a, b = src.location, dst.location

        # Determine whether this is an asynchronous move
        if isasync(tensor_graph, src, dst)
            concurrent = src.op
        else
            concurrent = nothing
        end

        if write_to_pmem(a, b)
            if written_to_pmem
                @show actions
                error()
            end
            # All downstream users are consumers
            consumers = unique(vertices[i].op for i in i:length(vertices) if vertices[i].isuser)

            # One solution to the ILP is to move data back and forth if it does not cause
            # any additional overhead.
            #
            # Here, we just filter out these movements by checking if the consumers of a
            # move is empty
            if !isempty(consumers)
                push!(actions, MoveAction(consumers, PMEM, true, concurrent))
                written_to_pmem = true
            end
        end

        if read_from_pmem(a, b)
            consumers = getkeeps(vertices, i)
            if !isempty(consumers)
                push!(actions, MoveAction(consumers, DRAM, false, concurrent))
            end
        end
    end

    # Need to filter out the first op from showing up in the actions because the first op
    # doesn't actually use the tensor - it produces it.
    producing_op = first(vertices).op

    for action in actions
        filter!(!isequal(producing_op), action.consumers)
    end
    return actions
end
