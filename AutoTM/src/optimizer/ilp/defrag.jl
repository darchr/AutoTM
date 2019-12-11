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
# To deal with this, we take the ALL instances where the memory limit is exceeded due
# to fragmentation and reduce the DRAM limit for the nodes around this instance.
#
# This should cause the ngraph allocator to free up some space so we don't go over the
# limit.
offenders(frame::Frame) = offenders(frame.modeltype, frame.profile_data)
function offenders(I::ILPHolder, data::FunctionData, io_bytes)
    dram_limits = I.dram_limits
    ml = maxlimit(I)
    @show ml

    # Go through all of the live tensors - find the first that exceeds the limit
    offending_tensors = XTensor{XNode}[]
    offending_nodes = XNode[]
    worst = 0
    for (index, live) in enumerate(live_tensors(data))
        # Find the DRAM tensors
        dram_tensors = filter(!is_persistent, live)
        isempty(dram_tensors) && continue

        # Find all out of bounds tensors
        for tensor in dram_tensors
            sz = (io_bytes + nGraph.get_pool_offset(unx(tensor)) + sizeof(tensor)) / 1E6
            if sz > ml
                push!(offending_tensors, tensor)
                worst = max(worst, sz)
            end
        end

        # Check if we have a workspace and get the offset of the workspace
        node = nodes(data)[index]
        offset = nGraph.Lib.get_workspace_tensor_offset(nGraph.getpointer(unx(node)))
        tensor_size = nGraph.Lib.get_workspace_tensor_size(nGraph.getpointer(unx(node)))
        sz = (io_bytes + offset + tensor_size) / 1E6
        if sz > ml
            push!(offending_nodes, node)
            worst = max(worst, sz)
        end
    end

    return offending_tensors, offending_nodes, worst
end

function update(I::T, local_args, data::FunctionData) where {T <: ILPHolder}
    dram_limits = I.dram_limits
    io_bytes = isempty(local_args) ? 0 : sum(sizeof, local_args)
    ml = maxlimit(I)

    offending_tensors, offending_nodes, worst = offenders(I, data, io_bytes)

    @info "Allocation size: " worst

    decrease_amount = max(
        # Decrease by at most 5%
        0.95,
        # If the overuse is small, just decrease by a tiny amount
        1 - ((worst / ml) - 1) / 2,
    )

    # Keep track of the indices that need their limits lowered
    indices = Int[]
    for node in offending_nodes
        push!(indices, I.node_to_limit_index[nGraph.name(node)])
    end

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
        if checkbounds(Bool, dram_limits, idx)
            dram_limits[idx] = round(Int, decrease_amount * dram_limits[idx])
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

