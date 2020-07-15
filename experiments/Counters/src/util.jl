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
modify(f::Function) = string(last(split(string(f), ".")))

function make_params(f, nt::NamedTuple{names}) where {names}
    return NamedTuple{(:benchmark, names...)}(modify.((f, nt...)))
end

getnames(::Type{<:NamedTuple{names}}) where {names} = names

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
