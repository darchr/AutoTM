# Model with synchronous "move" nodes
#
# The formulation here is similar to the "Simple" formulation, but now data is allowed
# to move from DRAM to PMEM using synchronous move nodes. Using a move node adds
# execution time based on the relative read and write bandwidths.
#
# This is realized by generating a tensor location set for each op the tensor is live.
#
# We then create a variable that is "active" when a tensor changes location from DRAM
# to PMEM. These variables will add time to the objective function
struct TensorMeta
    graph::MetaGraph

    # Nodes using this tensor
    users::Vector{XNode}

    # Look-up a node wrapper, get the node that serves as a reference for this
    reference_map::Dict{XNode, XNode}
end

get_reference(S::TensorMeta, node::XNode) = S.reference_map[node]
graph(S::TensorMeta) = S.graph
Profiler.users(S::TensorMeta) = S.users

#####
##### Model Types
#####

# Singleton types for dispatching to different formulations
abstract type ILPFormulationType end
struct IsFixed <: ILPFormulationType end
struct IsSynchronous <: ILPFormulationType end
struct IsAsynchronous <: ILPFormulationType end

mutable struct ILPHolder{T <: ILPFormulationType}
    dram_limits::Vector{Int}

    descriptors::Dict{XTensor{XNode}, TensorMeta}
    async_move_vars::Dict{XNode, Vector{JuMP.VariableRef}}

    # Need to key this with string names instead of XNode because of recompiling the ngraph
    # function and reporifling.
    #
    # New XNodes get created that don't hash the same as the old ones.
    node_to_limit_index::Dict{String, Int}

    # Bandwidths
    read_bandwidth::Int64
    write_bandwidth::Int64
    read_bandwidth_async::Int64
    write_bandwidth_async::Int64

    # Flag to determine if we need to defrag
    defrag::Bool
end

# Add factory methods
exceeds_limit(fex::nGraph.FluxExecutable, I::ILPHolder, local_args) =
    exceeds_limit(fex.ex.ngraph_function, I)

function exceeds_limit(f::nGraph.NFunction, I::ILPHolder, local_args)
    I.defrag || return false

    # Get the allocated bytes, including local arguments
    io_bytes = isempty(local_args) ? 0 : sum(sizeof, local_args)
    @info "IO Bytes: " io_bytes
    @info "Lenght Local Args: " length(local_args)
    @info "Temporary Pool Size: " convert(Int, nGraph.get_temporary_pool_size(f))

    alloc_bytes = io_bytes + nGraph.get_temporary_pool_size(f)

    # Convert to MB and compare to the maximum limit
    return (alloc_bytes / 1E6) > maxlimit(I)
end

# The general idea is that heap fragmentation causes the actual allocated amount to
# exceed the limit.
#
# To deal with this, we take the FIRST instance where the memory limit is exceeded due
# to fragmentation and reduce the DRAM limit for the node just BEFORE that instance.
#
# This should cause the ngraph allocator to free up some space so we don't go over the
# limit.
offenders(frame::Frame) = offenders(frame.modeltype, frame.profile_data)
function offenders(I::ILPHolder, data::FunctionData, io_bytes = 0)
    dram_limits = I.dram_limits
    ml = maxlimit(I)

    # Go through all of the live tensors - find the first that exceeds the limit
    offending_tensors = XTensor{XNode}[]
    worst = 0
    for live in live_tensors(data)
        # Find the DRAM tensors
        dram_tensors = filter(!is_persistent, live)
        isempty(dram_tensors) && continue

        # Find all out of bounds tensors
        for tensor in dram_tensors
            sz = io_bytes + (nGraph.get_pool_offset(unx(tensor)) + sizeof(tensor)) / 1E6
            if sz > ml
                push!(offending_tensors, tensor)
                worst = max(worst, sz)
            end
        end
    end

    return offending_tensors, worst
end

function update(I::T, local_args, data::FunctionData) where {T <: ILPHolder}
    dram_limits = I.dram_limits
    io_bytes = isempty(local_args) ? 0 : sum(sizeof, local_args)
    ml = maxlimit(I)

    offending_tensors, worst = offenders(I, data, io_bytes)

    @info "Allocation size: " worst

    decrease_amount = max(
        # Decrease by at most 1%
        0.99,
        # If the overuse is small, just decrease by a tiny amount
        1 - ((worst / ml) - 1) / 2,
    )

    # Keep track of the indices that need their limits lowered
    indices = Int[]

    for tensor in offending_tensors
        for node in users(tensor)
            (ismove(unx(node)) || !hasprofile(node)) && continue
            push!(indices, I.node_to_limit_index[nGraph.name(node)])
        end
    end

    radius = 5
    # Expand indices around the radius
    indices = Iterators.flatten([(idx - radius):(idx + radius) for idx in unique(indices)]) |>
        collect |>
        unique
    for idx in indices
        # Scale surrounding regions as well
        for i in (idx - radius):(idx + radius)
            if checkbounds(Bool, dram_limits, i)
                dram_limits[i] = round(Int, decrease_amount * dram_limits[i])
            end
        end
    end

    # Return a new ILHolder
    return T(
        dram_limits,
        Dict{XTensor{XNode}, TensorMeta}(),
        Dict{XNode, Vector{JuMP.VariableRef}}(),
        Dict{String, Int}(),
        rb(I),
        wb(I),
        rba(I),
        wba(I),
        I.defrag,
    )
end

# Accessor methods
rb(I::ILPHolder) = I.read_bandwidth
wb(I::ILPHolder) = I.write_bandwidth
rba(I::ILPHolder) = I.read_bandwidth_async
wba(I::ILPHolder) = I.write_bandwidth_async

static(dram_limits; defrag = false) = ILPHolder{IsFixed}(
    dram_limits,
    Dict{XTensor{XNode}, TensorMeta}(),
    Dict{XNode, Vector{JuMP.VariableRef}}(),
    Dict{String, Int}(),
    1,1,1,1,
    defrag,
)

synchronous(dram_limits, a, b; defrag = false) = ILPHolder{IsSynchronous}(
    dram_limits,
    Dict{XTensor{XNode}, TensorMeta}(),
    Dict{XNode, Vector{JuMP.VariableRef}}(),
    Dict{String, Int}(),
    a,b,1,1,
    defrag,
)

asynchronous(dram_limits,a,b,c,d; defrag = false) = ILPHolder{IsAsynchronous}(
    dram_limits,
    Dict{XTensor{XNode}, TensorMeta}(),
    Dict{XNode, Vector{JuMP.VariableRef}}(),
    Dict{String, Int}(),
    a,b,c,d,
    defrag,
)

# Common Methods

# Length check because ngraph compilation is not 100 % consistent and can sometimes have
# a few more nodes than it began with ...
limit(F::Frame, args...) = limit(F.modeltype, args...)
limit(S::ILPHolder, i) = i > length(S.dram_limits) ? maxlimit(S) : S.dram_limits[i]
maxlimit(S::ILPHolder) = maximum(S.dram_limits)

predict(F::Frame) = objective_value(F.model)
descriptor(F::Frame, tensor::XTensor) = F.modeltype.descriptors[tensor]

#####
##### Entry Point
#####

# For procedurally building JuMP expressions for the ILP model
const _expr_type = typeof(AffExpr())

function create_model(modeltype::ILPHolder, profile_data::FunctionData)
    preprocess!(modeltype, profile_data)

    # Start with an empty model that we will progressively build.
    model = Model(with_optimizer(Gurobi.Optimizer; TimeLimit = 3600, MIPGap = 0.01))
    frame = Frame(modeltype, model, profile_data)

    # Going deep into JuMP here - the idea is to build the objective as a bunch of aff exprs
    # and eventually combine all of them together.
    model[:node_times] = Dict{String, _expr_type}()
    model[:tensor_async] = Dict{String, _expr_type}()
    model[:tensor_sync] = _expr_type()

    add_tensors!(frame)
    add_nodes!(frame)
    add_constraints!(frame)

    # Default objective is to just sum all of the node times
    objective_expr = model[:tensor_sync]
    for (node_name, node_times) in model[:node_times]
        # Check to see if there are overlapping async transfers.
        #
        # If so, take the max of the sum of the overlapping transfers and the node time.
        _async = get(model[:tensor_async], node_name, nothing)
        if isnothing(_async)
            add_to_expression!(objective_expr, node_times)
        else
            var = @variable(model, integer = true, lower_bound = 0)
            @constraint(model, var >= node_times)
            @constraint(model, var >= _async)
            add_to_expression!(objective_expr, var)
        end
    end
    # Quick optimization to remove zero terms
    drop_zeros!(objective_expr)
    @objective(frame.model, Min, objective_expr)

    return frame
end

## Metadata For graph creation
@enum VertexLocation LOC_PMEM LOC_DRAM LOC_DRAM_PRE LOC_SOURCE LOC_SINK

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
struct VertexMetadata
    # The gadget that this vertex belongs to. Used for edge generation.
    gadget::Int
    # The op index that this gadget refers to
    op::XNode
    # Where the vertex lives
    location::VertexLocation
    # What type of moves this vertex allows
    move_type::MoveType
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

# Preprocessing basically involves creating the tensor graphs for each intermediate tensor.
function _liverange(data::FunctionData, t::XTensor)
    start = findfirst(isequal(producer(t)), nodes(data))::Int
    stop = findlast(isequal(consumer(t)), nodes(data))
    isnothing(stop) && (stop = length(nodes))
    return start:stop
end

function _getgadgets(A::ILPHolder{IsAsynchronous}, data::FunctionData, t::XTensor)
    liverange = _liverange(data, t)
    livenodes = (nodes(data)[x] for x in liverange)
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
    for ind in liverange
        node = nodes(data)[ind]
        if in(node, users(t))
            push!(refs, (node = node, move_type = MOVE_SYNC))
            ref = node

        # Check if there is a user node within `bound`. If so, make this an async move node.
        elseif hasprofile(node) && !Utils.is_memory_intensive(unx(node)) &&
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

function _getgadgets(::ILPHolder{IsSynchronous}, data::FunctionData, t::XTensor)
    liverange = _liverange(data, t)
    livenodes = (nodes(data)[x] for x in liverange)

    # Build the referece map
    reference_map = Dict{XNode, XNode}()
    ref = producer(t)
    for ind in liverange
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

function _getgadgets(::ILPHolder{IsFixed}, data::FunctionData, t::XTensor)
    liverange = _liverange(data, t)
    producer = nodes(data)[first(liverange)]

    reference_map = Dict{XNode, XNode}()
    for ind in liverange
        reference_map[nodes(data)[ind]] = producer
    end

    return [(node = producer, move_type = MOVE_NONE)], reference_map
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
        # Get two things from _getgadgets:
        #
        # 1. A named tuple (node::XNode, move_type::MoveType)
        # 2. A dictionary implementing the `ref` function.
        gadgets, reference_map = _getgadgets(S, data, tensor)

        @assert !isempty(gadgets)

        # Graph building time :D
        g = MetaGraph(DiGraph(), EdgeMetadata, VertexMetadata)

        # Get the users so we can annotate if a gadget node is a user

        # Add nodes for each region
        for (count, nt) in enumerate(gadgets)
            # Unpacek the
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

        # Use a quadratic complexity algorithm for doing edge assignment. It's not
        # perfect but it's simple, and as long as the graphs don't get too big should
        # run quickly enough for our purposes.
        for src in vertices(g), dst in vertices(g)
            src == dst && continue

            src_meta = _meta(g, src)
            dst_meta = _meta(g, dst)

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
        S.descriptors[tensor] = TensorMeta(g, [g.node for g in gadgets], reference_map)
    end
end

#####
##### Adding Tensors
#####

function add_tensors!(frame::Frame)
    data = frame.profile_data
    modeltype = frame.modeltype

    # Create variables for the tensors and add flow constraints to the to the tensor graphs
    @variable(frame.model,
        tensor_graphs[
            tensor = tensors(data),
            e = edges(graph(descriptor(frame, tensor)))
        ],
        Bin
    )

    @showprogress 1 "Creating Flow Formulation " for tensor in tensors(data)
        g = graph(descriptor(frame, tensor))
        # Iterate through nodes in the graph - generating constraints based on the type
        # of node.
        for v in vertices(g)
            # Set flow coming out of the source node
            if _meta(g, v).location == LOC_SOURCE
                @constraint(frame.model,
                    sum(tensor_graphs[tensor, e] for e in outedges(g, v)) == 1
                )

            # Set flow going into the sink node
            elseif _meta(g, v).location == LOC_SINK
                @constraint(frame.model,
                    sum(tensor_graphs[tensor, e] for e in inedges(g, v)) == 1
                )

            # All other ops must conserve flow
            else
                oe = collect(outedges(g, v))
                ie = collect(inedges(g, v))
                @constraint(frame.model,
                    sum(tensor_graphs[tensor, e] for e in oe) -
                    sum(tensor_graphs[tensor, e] for e in ie) == 0
                )
            end
        end
    end

    #####
    ##### Add objective penalty for moving data
    #####

    add_movement_formulations!(frame)

    #####
    ##### Create variables to determine if a tensor is in DRAM.
    #####

    @variable(frame.model,
        tensor_in_dram[
            tensor = tensors(data),
            user = nGraph.name.(users(descriptor(frame, tensor)))
        ],
        Bin
    )

    @variable(frame.model,
        tensor_in_dram_post[
            tensor = tensors(data),
            user = nGraph.name.(users(descriptor(frame, tensor)))
        ],
        Bin
    )

    # A tensor in DRAM is live if any of its incoming edges are used.
    for tensor in tensors(data)
        desc = descriptor(frame, tensor)
        g = graph(desc)

        # Create a container for the critical edge from LOC_SOURCE to LOC_SINK if this is
        # an argument tensor
        skip_edge = []
        if isarg(tensor) 
            # Find the source to sink edge.
            # Will error if this edge doesn't exist, so serves as a form of error checking
            edge = find_edge(g,
                (g, e) -> _meta(g, src(e)).location == LOC_SOURCE && 
                          _meta(g, dst(e)).location == LOC_SINK
            )
            push!(skip_edge, edge)
        end

        for user in users(desc)
            # Get the DRAM for this op.
            verts = filter(
                v -> isdram(_meta(g,v).location) && _meta(g,v).op == user,
                vertices(g)
            )

            # Map `inedges` to `vertex_iter` and iterats over all those edges
            _iter = vflatten(Iterators.flatten(inedges.(Ref(g), verts)), skip_edge)
            for e in _iter
                @constraint(
                    frame.model,
                    tensor_in_dram[tensor, nGraph.name(user)] >= tensor_graphs[tensor, e]
                )
            end

            # If all incoming edges are not taken, tensor MUST not be in DRAM.
            @constraint(frame.model,
                sum(tensor_graphs[tensor, e] for e in _iter) >=
                    tensor_in_dram[tensor, nGraph.name(user)]
            )

            # Similary, set the post DRAM constraints
            _edges = filter(
                e -> (isdram(_meta(g, src(e)).location) &&
                    # Need to check "LOC_SINK" for the static case
                    (
                        isdram(_meta(g, dst(e)).location) ||
                        _meta(g, dst(e)).location == LOC_SINK
                    ) && _meta(g, src(e)).op == user),
                collect(edges(g))
            )
            _iter = vflatten(_edges, skip_edge)
            for e in _iter
                @constraint(
                    frame.model,
                    tensor_in_dram_post[tensor, nGraph.name(user)] >= tensor_graphs[tensor, e]
                )
            end

            @constraint(frame.model,
                sum(tensor_graphs[tensor, e] for e in _iter) >=
                    tensor_in_dram_post[tensor, nGraph.name(user)]
            )
        end
    end

    return nothing
end

# Filter on edge type, sort by parent index to get the edges in execution order.
_find_edges(g, edgetype) = sort(
    filter(e -> _meta(g, e).edgetype == edgetype, collect(edges(g))),
    by = src
)

# Formulation specific move node stuff
add_movement_formulations!(frame::Frame{ILPHolder{IsFixed}}) = nothing
function add_movement_formulations!(frame::Frame)
    # Unpack variables
    data = frame.profile_data
    modeltype = frame.modeltype

    tensor_sync_expr = frame.model[:tensor_sync]
    tensor_async_dict = frame.model[:tensor_async]
    tensor_graphs = frame.model[:tensor_graphs]

    # A tensor is written to dram if:
    # - It was not created into PMEM
    # - Any edge from DRAM to PMEM is taken
    #
    # NOTE: We only pay the write cost once.
    @variable(frame.model, tensor_write[tensor = tensors(data)], Bin)

    # Add objective terms for all read ops
    for tensor in tensors(data)
        # Skip if this tensor can never be assigned to PMEM
        in(PMEM, locations(tensor)) || continue

        # Some unpacking
        g = graph(descriptor(frame, tensor))
        bytes = sizeof(tensor)

        # Take the ceiling of all these to ensure there's always a cost to moving.
        read_cost = ceil(Int, bytes / rb(modeltype))
        write_cost = ceil(Int, bytes / wb(modeltype))
        read_cost_async = ceil(Int, bytes / rba(modeltype))
        write_cost_async = ceil(Int, bytes / wba(modeltype))

        # Collect Edges according to type.
        sync_reads = _find_edges(g, EDGE_SYNC_READ)
        sync_writes = _find_edges(g, EDGE_SYNC_WRITE)
        async_reads = _find_edges(g, EDGE_ASYNC_READ)
        async_writes = _find_edges(g, EDGE_ASYNC_WRITE)

        #####
        ##### Constraints for sync write variable
        #####

        # No Sync write if any async write
        for e in sync_writes
            @constraint(
                frame.model,
                tensor_write[tensor] >= tensor_graphs[tensor, e]
            )
        end

        # No sync write if no edges taken
        @constraint(
            frame.model,
            tensor_write[tensor] <= sum(tensor_graphs[tensor, e] for e in sync_writes)
        )

        #####
        ##### Constraints on async write variables
        #####

        # Assign each edge to the kernel it overlaps with.
        #
        # Just go overkill and grab all the edges, even though we end up only using a subset.
        kernels = Dict(e => _meta(g,src(e)).op for e in edges(g))

        # Create read variables expressions
        for e in async_reads
            _expr = get!(tensor_async_dict, nGraph.name(kernels[e]), _expr_type())
            move_var = tensor_graphs[tensor, e]
            add_to_expression!(_expr, read_cost_async, move_var)
            dict_push!(frame.modeltype.async_move_vars, kernels[e], move_var)
        end

        # Create write variables.
        for e in async_writes
            _expr = get!(tensor_async_dict, nGraph.name(kernels[e]), _expr_type())
            move_var = tensor_graphs[tensor, e]
            add_to_expression!(_expr, write_cost_async, move_var)
            dict_push!(frame.modeltype.async_move_vars, kernels[e], move_var)
        end

        #####
        ##### Finally, add all the synchonous move costs.
        #####
        for e in sync_reads
            add_to_expression!(tensor_sync_expr, read_cost, tensor_graphs[tensor, e])
        end
        add_to_expression!(tensor_sync_expr, write_cost, tensor_write[tensor])
    end
    return nothing
end

# There's an issue when trying to reference whether or not a tensor is in DRAM.
#
# If we're on an op where the tensor is used, we have to look at the inputs to a
# graph verted with LOC_DRAM or LOC_PREAD to see if the tensor was fetched or already
# lived in dram.
#
# If we're on an op where a tensor is LIVE but not READ, we need to check the outgoing
# edge of the correct DRAM -> DRAM node to see if the tensor just lives around in DRAM.
function get_tensor_in_dram(F::Frame, tensor::XTensor, node::XNode)
    desc = descriptor(F, tensor)
    if in(node, users(desc))
        return F.model[:tensor_in_dram][tensor, nGraph.name(node)]
    else
        return F.model[:tensor_in_dram_post][tensor, nGraph.name(get_reference(desc, node))]
    end
end

function add_nodes!(F::Frame)
    data = F.profile_data

    # Create decision variables for all nodes that have a choice of backend algorithm.
    # TODO: reimplement
    select_nodes = filter(x -> hasprofile(x) && can_select_algo(x), nodes(data))
    if !isempty(select_nodes)
        @info "Creating Algorithms Variables"
        @variable(
            F.model,
            algo_var[
                node = select_nodes,
                enum = enums(gettime(node))
            ],
            Bin
        )

        # Constrain so only one algorithm may be selected.
        for node in select_nodes
            @constraint(
                F.model,
                sum(algo_var[node, e] for e in enums(gettime(node))) == 1
            )
        end
    end

    for node in nodes(data)
        # We don't profile all ops, so perform a quick check to see if this is an op
        # the we have profile information for. If not, there's nothing to do as far as the
        # ILP model is concerned.
        hasprofile(node) || continue

        # The GPU path of this code will just return an all DRAM config - which will be
        # useful for generating the constraint that all kernel IO for the GPU case must
        # reside in GPU DRAM.
        #
        # The CPU path will yield a bunch of DRAM/PMEM combinations
        configs = configs_for(node)

        # Create a variable for each config.
        vars = @variable(F.model, [config = configs], Bin)

        for config in configs
            # Create an expression for the input and output locations
            expr = _expr_type()
            iter = Iterators.flatten((
                zip(config.inputs, inputs(node)),
                zip(config.outputs, outputs(node))
            ))

            for (location, tensor) in iter
                # use `jump_tensor` because it's really a JuMP variable that is returned
                # by this call.
                jump_tensor = get_tensor_in_dram(F, tensor, node)
                if location == DRAM
                    add_to_expression!(expr, jump_tensor)
                    @constraint(F.model, vars[config] <= jump_tensor)
                else
                    add_to_expression!(expr, 1)
                    add_to_expression!(expr, -1, jump_tensor)
                    @constraint(F.model, vars[config] <= 1 - jump_tensor)
                end
            end

            @constraint(
                F.model,
                vars[config] + length(config.inputs) + length(config.outputs) >= 1 + expr
            )
        end

        # Add a valid contraint to help the solver
        @constraint(F.model, sum(vars[config] for config in configs) == 1)

        # Create an expression for this node's expected running time.
        node_times = _expr_type()
        for config in configs
            # If we can select the algorithm for this node, we need to generate some more
            # variables to "AND" the config with the algorithm selection to ensure that
            # we only get a single algorithm out at the end.
            #
            # If there are not multiple algorithms, then we don't have to worry about it.
            if can_select_algo(node, config)
                v = @variable(F.model, [enum = enums(gettime(node, config))], Bin)
                for enum in enums(gettime(node, config))
                    @constraint(F.model, v[enum] <= algo_var[node, enum])
                    @constraint(F.model, v[enum] <= vars[config])
                    @constraint(F.model, v[enum] + 1 >= vars[config] + algo_var[node, enum])

                    coeff = ceil(Int64, Profiler.timeat(gettime(node, config), enum))
                    add_to_expression!(node_times, coeff, v[enum])
                end
            else
                coeff = ceil(Int64, gettime(node, config))
                add_to_expression!(node_times, coeff, vars[config])
            end
        end
        F.model[:node_times][nGraph.name(node)] = node_times
    end
    return nothing
end

# Allocations in ngraph happen on 4096 bytes boundaries. For better accuracty, round
# up to the nearest multiple of 4096 before figuring out the number of bytes.
#
# Take the floor to introduce more zeros into the ILP formulation. This shouldn't really
# make much of a difference.
tensor_size(t::XTensor) = tensor_size(sizeof(t))
tensor_size(sz) = ceil(Int, ceil(Int, sz / 4096) * 4096 / 1E6)

function add_constraints!(F::Frame)
    # Unpack some variables
    data = F.profile_data

    for (index, tensors) in enumerate(live_tensors(data))
        node = nodes(data)[index]
        hasprofile(node) || continue
        F.modeltype.node_to_limit_index[nGraph.name(node)] = index

        # Add DRAM constraint for the workspace
        if can_select_algo(node)
            v = F.model[:algo_var]
            algo_expr = @expression(
                F.model,
                sum(
                    tensor_size(Profiler.bytesat(gettime(node), e)) *
                    v[node, e] for e in enums(gettime(node))
                )
            )
        else
            # If we can't select the algorithm, just create an empty expression that will
            # be hapily optimized away when we create the size constraint.
            algo_expr = _expr_type()
        end

        if !isempty(tensors)
            @constraint(F.model,
                algo_expr + sum(tensor_size(t) * get_tensor_in_dram(F, t, node)
                    for t in tensors
                    if !iszero(tensor_size(t))) <= limit(F, index)
            )
        end
    end

    return nothing
end

#####
##### Queries
#####

# Check if a given tensor is:
# 1. A function argument (nGraph function input or output)
# 2. Assigned to DRAM for the lifetime of the function
function islocalarg(frame, tensor::XTensor)
    # Determine if it's an argument
    isarg(tensor) || return false

    # Find the critical edge from LOC_SOURCE to LOC_SINK
    # This tensor is assigned to the local memory if this edge is taken (has a solved value
    # of one)
    g = descriptor(frame, tensor).graph
    edge = find_edge(g,
        (g, e) -> _meta(g, src(e)).location == LOC_SOURCE && _meta(g, dst(e)).location == LOC_SINK
    )
     
    return approx_one(frame.model[:tensor_graphs][tensor, edge])
end

