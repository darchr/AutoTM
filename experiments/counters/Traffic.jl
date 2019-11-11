module Traffic

using SIMD
using Random

# Processing pipeline for name formatting
modify(x) = x
modify(s::Union{String,Symbol}) = replace(String(s), "-"=>"_")
modify(::Val{T}) where {T} = T

stripmod(f::Function) = last(split(string(f), "."))

modify(x, y) = "$(modify(x))=$(modify(y))"

function make_filename(f, nt::NamedTuple; dir = "array_data")
    kvs = (modify(k, v) for (k, v) in pairs(nt))
    return "$(dir)/$(stripmod(f))_$(join(kvs, "_")).jls"
end

function vector_sum(
        A::AbstractArray{T},
        ::Val{N},
        aligned = Val(false),
        nontemporal = Val(false)
    ) where {T, N}

    unroll = 4

    # Make sure we can successfully chunk up this array into SIMD sizes
    @assert iszero(mod(length(A), unroll * N))

    # If we've passed alignment flags, make sure the base pointer of this array is in fact
    # aligned correctly.
    if aligned == Val{true}()
        @assert iszero(mod(Int(pointer(A)), sizeof(T) * N))
    end

    s = Vec{N,T}(zero(T))
    @inbounds for i in 1:(unroll*N):length(A)
        _v1 = vload(Vec{N,T}, pointer(A, i),         aligned, nontemporal)
        _v2 = vload(Vec{N,T}, pointer(A, i + N),     aligned, nontemporal)
        _v3 = vload(Vec{N,T}, pointer(A, i + 2 * N), aligned, nontemporal)
        _v4 = vload(Vec{N,T}, pointer(A, i + 3 * N), aligned, nontemporal)

        s += _v1 + _v2 + _v3 + _v4
    end
    return sum(s)
end

function vector_write(
        A::AbstractArray{T},
        ::Val{N},
        aligned = Val(false),
        nontemporal = Val(false),
    ) where {T, N}

    unroll = 4

    # Make sure we can successfully chunk up this array into SIMD sizes
    @assert iszero(mod(length(A), unroll * N))

    # If we've passed alignment flags, make sure the base pointer of this array is in fact
    # aligned correctly.
    if aligned == Val{true}()
        @assert iszero(mod(Int(pointer(A)), sizeof(T) * N))
    end

    # We need to do some schenanigans to get LLVM to emit the correct code.
    # Just doing something like
    #    vstore(s, pointer(A, i),         aligned, nontemporal)
    #    vstore(s, pointer(A, i + N),     aligned, nontemporal)
    #    vstore(s, pointer(A, i + 2 * N), aligned, nontemporal)
    #    vstore(s, pointer(A, i + 3 * N), aligned, nontemporal)
    #
    # Results in spurious `mov` instructions between the vector stores for pointer
    # conversions, even though these are really not needed.
    #
    # Instead, we perform the pointer arithmetic manually.
    s = Vec{N,T}(zero(T))
    base = pointer(A)
    @inbounds for i in 0:(unroll*N):(length(A) - 1)
        vstore(s, base + sizeof(T) * i,           aligned, nontemporal)
        vstore(s, base + sizeof(T) * (i + N),     aligned, nontemporal)
        vstore(s, base + sizeof(T) * (i + (2*N)), aligned, nontemporal)
        vstore(s, base + sizeof(T) * (i + (3*N)), aligned, nontemporal)
    end
    return nothing
end

function vector_increment(
        A::AbstractArray{T},
        ::Val{N},
        aligned = Val(false),
        nontemporal = Val(false),
    ) where {T, N}

    unroll = 4

    # Make sure we can successfully chunk up this array into SIMD sizes
    @assert iszero(mod(length(A), unroll * N))

    # If we've passed alignment flags, make sure the base pointer of this array is in fact
    # aligned correctly.
    if aligned == Val{true}()
        @assert iszero(mod(Int(pointer(A)), sizeof(T) * N))
    end

    s = Vec{N,T}(one(T))
    base = pointer(A)
    @inbounds for i in 0:(unroll*N):(length(A) - 1)
        _v1 = vload(Vec{N,T}, base + sizeof(T) * i,           aligned, nontemporal)
        _v2 = vload(Vec{N,T}, base + sizeof(T) * (i + N),     aligned, nontemporal)
        _v3 = vload(Vec{N,T}, base + sizeof(T) * (i + (2*N)), aligned, nontemporal)
        _v4 = vload(Vec{N,T}, base + sizeof(T) * (i + (3*N)), aligned, nontemporal)

        _u1 = _v1 + s
        _u2 = _v2 + s
        _u3 = _v3 + s
        _u4 = _v4 + s

        vstore(_u1, base + sizeof(T) * i,           aligned, nontemporal)
        vstore(_u2, base + sizeof(T) * (i + N),     aligned, nontemporal)
        vstore(_u3, base + sizeof(T) * (i + (2*N)), aligned, nontemporal)
        vstore(_u4, base + sizeof(T) * (i + (3*N)), aligned, nontemporal)
    end
    return nothing
end

# Linear Feedback Shift Register
struct LFSR{D}
    feedback::Int
end
modify(::LFSR{D}) where {D} = D

Base.length(::LFSR{D}) where {D} = (2^D) - 1

# Implement LFSRs for some sizes.
# Coefficients are taken from https://users.ece.cmu.edu/~koopman/lfsr/index.html
# Becuase of our implementation, we have to chop off the MSB
const FEEDBACK = Dict(
     8 => 0x8E,
    25 => 0x1000004,
    26 => 0x2000023,
    27 => 0x4000013,
    28 => 0x8000004,
    29 => 0x8000004,
    30 => 0x20000029,
    31 => 0x40000004,
    32 => 0x80000057,
    33 => 0x100000029,
    34 => 0x200000073,
)

@inline Base.iterate(::LFSR) = (1, 1)
@inline function Base.iterate(L::LFSR, previous)
    feedback = isodd(previous) ? L.feedback : 0
    next = xor(previous >> 1, feedback)

    # If the new term is 1, we're back where we started, so abort
    isone(next) && return nothing
    return next, next
end

function hop_sum(A::AbstractArray{T}, ::Val{N}, lfsr) where {T,N}
    return _hop_sum(reinterpret(Vec{N,T}, A), lfsr)
end

@inline function _hop_sum(A::AbstractArray{Vec{N,T}}, lfsr) where {N,T}
    s = zero(eltype(A))
    base = Base.unsafe_convert(Ptr{T}, pointer(A))
    @inbounds for i in lfsr
        ptr = base + sizeof(eltype(A)) * (i-1)
        v = vload(Vec{N,T}, ptr, Val{true}())
        s += v
    end
    return s
end

function hop_write(A::AbstractArray{T}, ::Val{N}, lfsr) where {T,N}
    return _hop_write(reinterpret(Vec{N,T}, A), lfsr)
end

@inline function _hop_write(A::AbstractArray{Vec{N,T}}, lfsr) where {N,T}
    s = zero(eltype(A))
    base = Base.unsafe_convert(Ptr{T}, pointer(A))
    @inbounds for i in lfsr
        ptr = base + sizeof(eltype(A)) * (i-1)
        vstore(s, ptr, Val{true}())
    end
    return nothing
end

function hop_increment(A::AbstractArray{T}, ::Val{N}, lfsr) where {T,N}
    return _hop_increment(reinterpret(Vec{N,T}, A), lfsr)
end

@inline function _hop_increment(A::AbstractArray{Vec{N,T}}, lfsr) where {N,T}
    base = Base.unsafe_convert(Ptr{T}, pointer(A))
    @inbounds for i in lfsr
        ptr = base + sizeof(eltype(A)) * (i-1)
        v = vload(Vec{N,T}, ptr, Val{true}())
        vstore(v + one(eltype(A)), ptr, Val{true}())
    end
end

end # module
