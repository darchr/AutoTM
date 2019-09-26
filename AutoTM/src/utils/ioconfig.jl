@enum TensorLocation::UInt8 DRAM PMEM

"""
Location information for the input and output tensors of a node.
"""
struct IOConfig{N, M}
    inputs::NTuple{N, TensorLocation}
    outputs::NTuple{M, TensorLocation}
end

Base.iterate(io::IOConfig, args...) = iterate(Iterators.flatten((io.inputs, io.outputs)), args...)
Base.length(io::IOConfig{N,M}) where {N,M} = N + M
function Base.getindex(io::IOConfig{N,M}, idx::Integer) where {N,M}
    if idx <= N
        return io.inputs[idx]
    elseif idx <= length(io)
        return io.outputs[idx - N]
    else
        throw(BoundsError(io, idx))
    end
end

Base.getindex(io::IOConfig, I::Vector{Int}) = [io[i] for i in I]

function setindex(io::IOConfig{N,M}, idx::Integer, x::TensorLocation) where {N,M}
    if idx <= N
        inputs = ntuple(i -> i == idx ? x : io.inputs[i], N)
        outputs = io.outputs
    elseif idx <= length(io)
        idx = idx - length(io.inputs)
        inputs = io.inputs
        outputs = ntuple(i -> i == idx ? x : io.outputs[i], M)
    end
    return IOConfig{N,M}(inputs, outputs)
end

function Base.isless(a::IOConfig{N,M}, b::IOConfig{N,M}) where {N,M}
    return (a.inputs < b.inputs) || ((a.inputs == b.inputs) && a.outputs < b.outputs)
end

function Base.show(io::IO, config::IOConfig{N,M}) where {N,M}
    f = x -> (x == DRAM) ? "DRAM" : "PMEM"
    print(io, "IOConfig{$N,$M}: ")
    print(io, "(", join(f.(config.inputs), ", "), ") -- ")
    print(io, "(", join(f.(config.outputs), ", "), ")")
end
