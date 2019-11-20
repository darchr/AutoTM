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

function paramtransfer(pipe, @nospecialize(x))
    serialize(PARAMPATH, x)
    println(pipe, "params")
    return nothing
end

# Processing pipeline for name formatting
modify(x) = x
modify(s::Union{String,Symbol}) = replace(String(s), "-"=>"_")
modify(::Val{T}) where {T} = T
modify(x::LFSR) = ceil(Int, log2(length(x)))

# If passed a function, get the name of the function
# If passed a string, don't do anything
stripmod(f::Function) = last(split(string(f), "."))
srtipmod(f::String) = f

function make_params(f, nt::NamedTuple{names}) where {names}
    return NamedTuple{(:benchmark, names...)}(modify.((f, nt...)))
end

getnames(::Type{<:NamedTuple{names}}) where {names} = names

#####
##### mmap a hugepage.
#####

const MAP_HUGETLB    = Cint(0x40000)
const MAP_HUGE_SHIFT = Cint(26)
const MAP_HUGE_2MB   = Cint(21 << MAP_HUGE_SHIFT)
const MAP_HUGE_1GB   = Cint(30 << MAP_HUGE_SHIFT)

abstract type AbstractPagesize end
struct Pagesize4K <: AbstractPagesize end
struct Pagesize2M <: AbstractPagesize end
struct Pagesize1G <: AbstractPagesize end

extraflags(::Pagesize4K) = Cint(0)
extraflags(::Pagesize2M) = MAP_HUGETLB | MAP_HUGE_2MB
extraflags(::Pagesize1G) = MAP_HUGETLB | MAP_HUGE_1GB

# Align length for `munmap` to a multiple of page size.
pagesize(::Pagesize4K) = 4096
pagesize(::Pagesize2M) = 2097152
pagesize(::Pagesize1G) = 1073741824

align(x, p::AbstractPagesize) = ceil(Int, x / pagesize(p)) * pagesize(p)
align(x, ::Pagesize4K) = x

# This is heavily based on the Mmap stdlib
function hugepage_mmap(::Type{T}, dim::Integer, pagesize::AbstractPagesize) where {T}
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
        systemerror(
            "munmap",
            ccall(:munmap, Cint, (Ptr{Cvoid}, Csize_t), ptr, align(mmaplen, pagesize)) != 0
        )
    end
    return A
end

#####
##### Custom Threading
#####

# Break up an array into a bunch of views and distribute them across threads.
#
# This makes sure that things are getting split up how we expect them to be.
function threadme(f, A, args...; prepare = false, iterations = 1)
    nthreads = Threads.nthreads()
    @assert iszero(mod(length(A), nthreads))

    step = div(length(A), nthreads)
    Threads.@threads for i in 1:Threads.nthreads()
        threadid = Threads.threadid()
        start = step * (i-1) + 1
        stop = step * i
        x = view(A, start:stop)

        # Run the inner loop
        for j in 1:iterations
            f(x, args...)
        end
    end
    return nothing
end
