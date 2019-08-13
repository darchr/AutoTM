module AllocatorModel

using IterTools

export  allocate,
        free,
        isfree,
        MemoryAllocator,
        MemoryNode,
        showfree

# moDNN Memory allocater
mutable struct MemoryNode
    isfree::Bool
    size::Int64
end
isfree(m::MemoryNode) = m.isfree
Base.sizeof(m::MemoryNode) = m.size

struct MemoryAllocator
    node_list::Vector{MemoryNode}
    alignment::Int64
end

MemoryAllocator(limit::Number, alignment::Number) = 
    MemoryAllocator(
        [MemoryNode(true, convert(Int, limit))], 
        convert(Int, alignment)
    )

function showfree(M::MemoryAllocator)
    offset = 0
    for node in M.node_list
        if isfree(node)
            @show (offset, sizeof(node))
        end
        offset += sizeof(node)
    end
    return nothing
end

function allocate(M::MemoryAllocator, sz)
    # Make the allocation size match the alignment
    sz = align(sz, M.alignment)
    offset = 0
    min_delta = typemax(Int64)
    best_offset = offset

    best_node = nothing
    best_index = zero(Int64)

    # Go through the list see if we can find the an exact fit
    for (index, node) in enumerate(M.node_list)
        if isfree(node) && sizeof(node) >= sz
            delta = sizeof(node) - sz
            if delta < min_delta
                min_delta   = delta
                best_node   = node
                best_offset = offset
                best_index  = index
            end
        end
        offset += sizeof(node)
    end

    # Check if we were able to allocate.
    isnothing(best_node) && return nothing

    # Do the allocation 
    if iszero(min_delta)  
        best_node.isfree = false
    else
        insert!(M.node_list, best_index, MemoryNode(false, sz))
        best_node.size -= sz
    end
    return best_offset
end

free(M::MemoryAllocator, ::Nothing) = nothing
function free(M::MemoryAllocator, offset)
    search_offset = 0
    found = false
    for node in M.node_list
        if (offset == search_offset)
            # Mark this node as free
            node.isfree = true
        end
        search_offset += sizeof(node)
    end

    # Coalesce free nodes together
    while true
        changed = false
        for (index, (a, b)) in enumerate(IterTools.partition(M.node_list, 2, 1))
            # If both nodes are free, merge the size into b and delete a
            #
            # `index` is the index of `a`.
            if isfree(a) && isfree(b) 
                b.size += sizeof(a)
                deleteat!(M.node_list, index)
                changed = true
                break
            end
        end
        changed || break
    end
    return nothing
end

function align(sz, alignment)
    if iszero(sz)
        sz = alignment
    else
        remainder = rem(sz, alignment)
        if remainder > zero(remainder)
            sz += (alignment - remainder)
        end
    end
    return sz
end

end
