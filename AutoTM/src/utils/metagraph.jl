"""
$(SIGNATURES)

Find the only vertex in `graph` satifsying `f`. Function `f` must take arguments `graph`
and `vertex` and return a boolean.

Return the vertex.
"""
function find_vertex(f, graph)
    idx = findonly(v -> f(graph, v), collect(LightGraphs.vertices(graph)))
    return idx
end

"""
$(SIGNATURES)

Find the only edge in `graph` satifsying `f`. Function `f` must take arguments `graph`
and `edge` and return a boolean.

Return the edge.
"""
function find_edge(f, graph)
    edges = collect(LightGraphs.edges(graph))
    idx = findonly(v -> f(graph, v), edges)
    return edges[idx]
end

# Why do these not exist in LightGraphs.jl??
inedges(g, v) = (LightGraphs.edgetype(g)(u, v) for u in LightGraphs.inneighbors(g, v))
outedges(g, v) = (LightGraphs.edgetype(g)(v, u) for u in LightGraphs.outneighbors(g, v))

"""
Custom graph type that allows associating metadata with graph edges and vertices.

$(METHODLIST)
"""
struct MetaGraph{T,E,V} <: LightGraphs.AbstractGraph{T}
    graph::LightGraphs.SimpleDiGraph{T}
    edge_meta::E
    vertex_meta::V

    """
    $(SIGNATURES)

    Construct a `MetaGraph` from `graph` with edge metadata type `E` and vertes metadata 
    type `V`.
    """
    function MetaGraph(graph::LightGraphs.SimpleDiGraph{T}, ::Type{E}, ::Type{V}) where {T,E,V}
        edge_meta = Dict{LightGraphs.edgetype(graph),E}()
        vertex_meta = Dict{eltype(graph),V}()
        return new{T,typeof(edge_meta),typeof(vertex_meta)}(graph, edge_meta, vertex_meta)
    end
end


const LIGHTGRAPHS_INTERFACE = (
    :(Base.reverse),
    :(LightGraphs.dst),
    :(LightGraphs.edges),
    :(LightGraphs.edgetype),
    :(LightGraphs.has_edge),
    :(LightGraphs.has_vertex),
    :(LightGraphs.inneighbors),
    :(LightGraphs.is_directed),
    :(LightGraphs.ne),
    :(LightGraphs.nv),
    :(LightGraphs.outneighbors),
    :(LightGraphs.src),
    :(LightGraphs.vertices),
)
for f in LIGHTGRAPHS_INTERFACE
    eval(:($f(M::MetaGraph, x...) = $f(M.graph, x...)))
end

LightGraphs.add_edge!(M::MetaGraph, src, dst) = LightGraphs.add_edge!(M.graph, src, dst)
function LightGraphs.add_edge!(M::MetaGraph, src, dst, metadata) 
    success = LightGraphs.add_edge!(M.graph, src, dst)
    success || error()

    M.edge_meta[LightGraphs.edgetype(M.graph)(src, dst)] = metadata
    return true
end

LightGraphs.add_vertex!(M::MetaGraph) = LightGraphs.add_vertex!(M.graph)
function LightGraphs.add_vertex!(M::MetaGraph, metadata)
    success = LightGraphs.add_vertex!(M.graph)
    success || error()

    M.vertex_meta[LightGraphs.nv(M)] = metadata
    return true
end

getmeta(M::MetaGraph, v::Integer) = M.vertex_meta[v]
getmeta(M::MetaGraph, e) = M.edge_meta[e]

function rem_edge!(M::MetaGraph, e)
    success = LightGraphs.rem_edge!(M.graph, e)
    if success
        delete!(M.edge_meta, e)
    end
    return success
end

