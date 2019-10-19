## Metadata For graph creation
@enum VertexLocation LOC_PMEM LOC_DRAM LOC_DRAM_PRE LOC_SOURCE LOC_SINK

# Helpers - generally pretty simple but helpful for abstracting the `isdram` attribute.
ispmem(loc::VertexLocation) = loc == LOC_PMEM
isdram(loc::VertexLocation) = loc == LOC_DRAM || loc == LOC_DRAM_PRE
issource(loc::VertexLocation) = loc == LOC_SOURCE
issink(loc::VertexLocation) = loc == LOC_SINK

@enum EdgeType begin
    EDGE_NONE
    EDGE_SYNC_READ
    EDGE_SYNC_WRITE
    EDGE_ASYNC_READ
    EDGE_ASYNC_WRITE
end

isasync(et::EdgeType) = in(et, (EDGE_ASYNC_READ, EDGE_ASYNC_WRITE))

@enum MoveType MOVE_NONE MOVE_SYNC MOVE_ASYNC

# Metadata to assign to each node in the liveness graph for tensors.
"""
    VertexMetadata

Metadata associated with the vertex of the tensor graph.

$(FIELDS)
"""
struct VertexMetadata
    "The gadget that this vertex belongs to. Used for edge generation."
    gadget::Int
    "The op index that this gadget refers to."
    op::XNode
    "Where the vertex lives."
    location::VertexLocation
    "What type of moves this vertex allows."
    move_type::MoveType
    "`true` is this vertex is a user of the tensor in whose graph this vertex appear."
    isuser::Bool
    vertex_number::Int
end

struct EdgeMetadata
    edgetype::EdgeType
end
isasync(em::EdgeMetadata) = isasync(em.edgetype)

#####
##### Preprocessing
#####

"""
$(SIGNATURES) -> UnitRange{Int64}

Return a range of indices corresponding to nodes during which `xtensor` is live.
"""
liverange(xtensor::XTensor) = (producer(xtensor).index):(consumer(xtensor).index)

function _getgadgets(A::ILPHolder{Asynchronous}, data::FunctionData, t::XTensor)
    range = liverange(t)
    livenodes = (nodes(data)[x] for x in range)
    refs = Vector{NamedTuple{(:node, :move_type),Tuple{XNode,MoveType}}}()

    # Build the referece map
    reference_map = Dict{XNode, XNode}()
    ref = producer(t)

    # To decide if a node should be considered as an aynchronous move point, we check to see
    # if the node is with `bound` distance of a user of the tensor.
    #
    # The intuition here is that moves will probably be located closer to their producers or
    # consumers rather than further.
    #
    # Making `bound` larger increased the search space of the formulation, which may lead to
    # better results at the cost of a larger mode.
    bound = 15
    move_time = sizeof(t) / A.write_bandwidth
    for ind in range
        node = nodes(data)[ind]
        if in(node, users(t))
            push!(refs, (node = node, move_type = MOVE_SYNC))
            ref = node

        # Check if there is a user node within `bound`. If so, make this an async move node.
        elseif hasprofile(node) &&
            any(
                in(users(t)),
                (nodes(data)[i] for i in max(ind-bound, 1):min(ind+bound, length(nodes(data))))
            )

            push!(refs, (node = node, move_type = MOVE_ASYNC))
            ref = node
        end
        reference_map[node] = ref
    end

    return refs, reference_map
end

function _getgadgets(::ILPHolder{Synchronous}, data::FunctionData, t::XTensor)
    range = liverange(t)
    livenodes = (nodes(data)[x] for x in range)

    # Build the referece map
    reference_map = Dict{XNode, XNode}()
    ref = producer(t)
    for ind in range
        node = nodes(data)[ind]
        if in(node, users(t))
            ref = node
        end
        reference_map[node] = ref
    end

    nt = [
        (node = u, move_type = isone(i) ? MOVE_NONE : MOVE_SYNC) for (i,u) in enumerate(users(t))
    ]

    return nt, reference_map
end

function _getgadgets(::ILPHolder{Static}, data::FunctionData, t::XTensor)
    range = liverange(t)
    producer = nodes(data)[first(range)]

    reference_map = Dict{XNode, XNode}()
    for ind in range
        reference_map[nodes(data)[ind]] = producer
    end

    return [(node = producer, move_type = MOVE_NONE)], reference_map
end

function getgadgets(x, data::FunctionData, t::XTensor)
    refs, reference_map = _getgadgets(x, data, t)
    # Update with argument
    
    # If this tensor `isarg`, then since it is live for the entire function, we need to 
    # insert node references for all nodes pointing to the producer.
    if isarg(t)
        ref = producer(t)
        for node in nodes(data)
            # Shortcut here, `get!` will lookup and automatically insert the mapping if it
            # doesn't exist.
            get!(reference_map, node, ref)
        end
    end

    return refs, reference_map
end

# TODO: When can't do anything because exhausted, make ASCII art for what's going on with
# these tensor graphs.
#=
=#

function edge_metadata(src, dst, s, d, src_move_type, tensor::XTensor)
    # Setup the correct move annotations.
    if src_move_type == MOVE_ASYNC
        edge_read_type = EDGE_ASYNC_READ
        edge_write_type = EDGE_ASYNC_WRITE
    else
        edge_read_type = EDGE_SYNC_READ
        edge_write_type = EDGE_SYNC_WRITE
    end

    # Determine if an edge should be added and what kind of edge it is.

    ### LOC_SOURCE node is source
    if (src, dst) == (LOC_SOURCE, LOC_DRAM_PRE)
        isone(d) && return EdgeMetadata(EDGE_NONE)
    elseif (src, dst) == (LOC_SOURCE, LOC_PMEM)
        isone(d) && return EdgeMetadata(EDGE_NONE)
    # If the tensor is an argument, we need to create an edge directly from source to sink.
    # This indicates that the tensor begins in DRAM - so will always be in DRAM
    elseif (src, dst) == (LOC_SOURCE, LOC_SINK) 
        isarg(tensor) && return EdgeMetadata(EDGE_NONE)

    ### LOC_DRAM as source

    # If we're in LOC_DRAM, data already exists in PMEM. Thus all edges originating in
    # LOC_DRAM have no metadata. I.E., no read or write type eges.
    elseif (src, dst) == (LOC_DRAM, LOC_DRAM)
        s == d-1 && return EdgeMetadata(EDGE_NONE)
    elseif (src, dst) == (LOC_DRAM, LOC_PMEM)
        s == d-1 && return EdgeMetadata(EDGE_NONE)
    elseif (src, dst) == (LOC_DRAM, LOC_SINK)
        s == d-1 && return EdgeMetadata(EDGE_NONE)

    ### LOC_DRAM_PRE as source

    # Moving from LOC_DRAM_PRE to LOC_PMEM indicates a movement of data from DRAM to PMEM
    # where the data did not exist in PMEM before. Thus, we must record a write.
    elseif (src, dst) == (LOC_DRAM_PRE, LOC_PMEM)
        s == d-1 && return EdgeMetadata(edge_write_type)
    # Movement between DRAM or to the SINK requires no metadata
    elseif (src, dst) == (LOC_DRAM_PRE, LOC_DRAM_PRE)
        s == d-1 && return EdgeMetadata(EDGE_NONE)
    elseif (src, dst) == (LOC_DRAM_PRE, LOC_SINK)
        s == d-1 && return EdgeMetadata(EDGE_NONE)

    ### LOC_PMEM as source
    
    # Movement edges from PMEM to DRAM. Must not happen on the first `component` because
    # prefetching from PMEM to DRAM at the time of creation makes no sense.
    elseif (src, dst) == (LOC_PMEM, LOC_DRAM)
        (s == d) && !isone(s) && return EdgeMetadata(edge_read_type)
    elseif (src, dst) == (LOC_PMEM, LOC_PMEM)
        (s == d-1) && return EdgeMetadata(EDGE_NONE)
    # Otherwise, no metadata needed
    elseif (src, dst) == (LOC_PMEM, LOC_SINK)
        (s == d-1) && return EdgeMetadata(EDGE_NONE)
    end
    return nothing
end

function preprocess!(S::ILPHolder, data::FunctionData)
     for tensor in tensors(data)

        # Get the users of this node
        # Get two things from getgadgets:
        #
        # 1. A named tuple (node::XNode, move_type::MoveType)
        # 2. A dictionary implementing the `ref` function.
        gadgets, reference_map = getgadgets(S, data, tensor)

        @assert !isempty(gadgets)

        # Graph building time :D
        g = MetaGraph(DiGraph(), EdgeMetadata, VertexMetadata)

        # If this tensor can only be assigned to a single location - don't generate a graph
        # for it - otherwise, populate the nodes in a tensor graph
        fixed_tensor = length(locations(tensor)) == 1
        if !fixed_tensor
            for (count, nt) in enumerate(gadgets)
                # Unpack the XNode
                node = nt.node
                move_type = nt.move_type
                islast = (count == length(gadgets))

                isuser = in(node, users(tensor))

                if count == 1
                    add_vertex!(g, VertexMetadata(0, node, LOC_SOURCE, move_type, isuser, nv(g)+1))
                end
                # Enumerate over locations that this tensor can live.
                #
                # Do it this way because some nodes can only live in DRAM, so iterating
                # then filtering takes care of that
                for location in locations(tensor)
                    if location == DRAM
                        # Add DRAM nodes
                        #
                        # Do not add these nodes if this tensor is an argument because it will
                        # have already originated in PMM
                        if !isarg(tensor) 
                            add_vertex!(g,
                                VertexMetadata(count, node, LOC_DRAM_PRE, move_type, isuser, nv(g)+1)
                            )
                        end

                        # only add a DRAM node if there could have been a write to PMEM
                        if count > 1
                            add_vertex!(g,
                                VertexMetadata(count, node, LOC_DRAM, move_type, isuser, nv(g)+1)
                            )
                        end
                    end

                    if location == PMEM
                        @assert !startswith(nGraph.name(tensor), "Constant")
                        # PMEM node
                        add_vertex!(g,
                            VertexMetadata(count, node, LOC_PMEM, move_type, isuser, nv(g)+1)
                        )
                    end
                end
                if islast
                    # Set the gadget number for the sink to one higher than the last count.
                    add_vertex!(g,
                        VertexMetadata(count + 1, node, LOC_SINK, move_type, isuser, nv(g)+1)
                    )
                end
            end
        end

        # Use a quadratic complexity algorithm for doing edge assignment. It's not
        # perfect but it's simple, and as long as the graphs don't get too big should
        # run quickly enough for our purposes.
        for src in vertices(g), dst in vertices(g)
            src == dst && continue

            src_meta = getmeta(g, src)
            dst_meta = getmeta(g, dst)

            metadata = edge_metadata(
                src_meta.location,
                dst_meta.location,
                src_meta.gadget,
                dst_meta.gadget,
                src_meta.move_type,
                tensor,
            )

            isnothing(metadata) && continue

            add_edge!(g, src, dst, metadata)
        end

        # Create the descriptor
        S.descriptors[tensor] = TensorMeta(
            g, 
            [g.node for g in gadgets], 
            reference_map,
            fixed_tensor
        )
    end
end

