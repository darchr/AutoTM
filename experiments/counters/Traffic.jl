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
struct LFSR{D,F} end
modify(::LFSR{D}) where {D} = D

Base.length(::LFSR{D}) where {D} = (2^D) - 1

# Implement LFSRs for some sizes.
# Coefficients are taken from https://users.ece.cmu.edu/~koopman/lfsr/index.html
# Becuase of our implementation, we have to chop off the MSB
_trunc(x, b) = convert(Int, x & ~(1 << (b-1)))
const FEEDBACK = Dict(
     8 => _trunc(0x8E, 8),
    25 => _trunc(0x1000004, 25),
    26 => _trunc(0x2000023, 26),
    27 => _trunc(0x4000013, 27)
    28 => _trunc(0x8000004, 28)
    29 => _trunc(0x8000004, 29),
    30 => _trunc(0x20000029, 30),
)

@inline Base.iterate(::LFSR) = (1, (1, 1))
@inline function Base.iterate(::LFSR{D,F}, state) where {D,F}
    previous = state[1]
    count =  state[2]
    count == ((1 << D) - 1) && return nothing

    lsb = previous & 1
    x = (previous >> 1) | (lsb << (D - 1))
    feedback = (lsb == 1) ? F : 0
    term = xor(x, feedback)
    return term, (term, count + 1)
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
    s = one(eltype(A))
    base = Base.unsafe_convert(Ptr{T}, pointer(A))
    @inbounds for i in lfsr
        ptr = base + sizeof(eltype(A)) * (i-1)
        v = vload(Vec{N,T}, ptr, Val{true}())
        vstore(v + s, ptr, Val{true}())
    end
end

end # module
