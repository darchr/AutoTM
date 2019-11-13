# Like serialize, but will also make a directory if needed.
function save(file, x)
    dir = dirname(file)
    isdir(dir) || mkpath(dir)
    serialize(file, x)
    return nothing
end

function transfer(pipe, @nospecialize(x))
    serialize(TRANSFERPATH, x)
    println(pipe, "measurements")
    return nothing
end

# Processing pipeline for name formatting
modify(x) = x
modify(s::Union{String,Symbol}) = replace(String(s), "-"=>"_")
modify(::Val{T}) where {T} = T

# If passed a function, get the name of the function
# If passed a string, don't do anything
stripmod(f::Function) = last(split(string(f), "."))
srtipmod(f::String) = f

# Create a nicely formatted
modify(x, y) = "$(modify(x))=$(modify(y))"

function make_filename(f, nt::NamedTuple; dir = "array_data")
    kvs = (modify(k, v) for (k, v) in pairs(nt))
    return "$(dir)/$(stripmod(f))_$(join(kvs, "_")).jls"
end

#####
##### Linear Feedback Shift Register
#####

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

#####
##### mmap a hugepage.
#####

const MAP_HUGETLB    = Cint(0x40000)
const MAP_HUGE_SHIFT = Cint(26)
const MAP_HUGE_2MB   = Cint(21 << MAP_HUGE_SHIFT)
const MAP_HUGE_1GB   = Cint(30 << MAP_HUGE_SHIFT)

abstract type AbstractPageSize end
struct Pagesize4K <: AbstractPageSize end
struct Pagesize2M <: AbstractPageSize end
struct Pagesize1G <: AbstractPageSize end

extraflags(::Pagesize4K) = Cint(0)
extraflags(::Pagesize2M) = MAP_HUGETLB | MAP_HUGE_2MB
extraflags(::Pagesize1G) = MAP_HUGETLB | MAP_HUGE_1GB

# This is heavily based on the Mmap stdlib
function hugepage_mmap(::Type{T}, dim::Integer, pagesize::AbstractPageSize) where {T}
    mmaplen = sizeof(T) * dim

    # Build the PROT flags - we want to be able to read and write.
    prot = Mmap.PROT_READ | Mmap.PROT_WRITE
    flags = Mmap.MAP_PRIVATE | Mmap.MAP_ANONYMOUS
    flags |= extraflags(pagesize)

    fd = Base.INVALID_OS_HANDLE
    offset = Cint(0)

    # Fordward this call into the Julia C library.
    ptr = ccall(
        :jl_mmap,
        Ptr{Cvoid},
        (Ptr{Cvoid}, Csize_t, Cint, Cint, RawFD, Int64),
        C_NULL,     # No address we really want.
        mmaplen,
        prot,
        flags,
        fd,
        offset,
    )

    # Wrap this into an Array and attach a finalizer that will unmap the underlying pointer
    # when the Array if GC'd
    A = Base.unsafe_wrap(Array, convert(Ptr{T}, UInt(ptr)), dim)
    finalizer(A) do x
        systemerror("munmap", ccall(:munmap, Cint, (Ptr{Cvoid}, Int), ptr, mmaplen) != 0)
    end
    return A
end
