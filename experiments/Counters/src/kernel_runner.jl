struct KernelParams
    # Size of the array for each thread.
    array_size::Int
    sampletime::Float64
    mode::String
    # The number of times to run each kernel. Larger kernels need to run for fewer
    # iterations.
    inner_iterations::Int
    # Callback for measurements.
    measurements::Vector{Any}
    vector_sizes::Vector{Int}
    temporals::Vector{Bool}
    array_type::String
end

function runkernels(params, f = (sz) -> Vector{Float32}(undef, sz))
    # Allocate and instantiate the array
    sz = Threads.nthreads() * (2 ^ params.array_size)
    @show sz
    A = f(sz)
    @show sizeof(A)
    threadme(vector_write, A, Val{16}())

    # Setup the counter
    pipe = connect(PIPEPATH)
    println(pipe, "sampletime $(params.sampletime)")

    # Do the benchmarking!
    benchmark(A, pipe, params)
end

function benchmark(A::AbstractArray, pipe, params::KernelParams)
    # Set up a bunch of nested loops to run through all the variations of tests we want
    # to execute.
    fns = [
    #    vector_sum,
    #    vector_write,
    #    vector_increment
    ]

    for (f, sz, _nt) in Iterators.product(fns, params.vector_sizes, params.temporals)
        # Create a NamedTuple of parameters to forward to the function.
        nt = (
            vector = Val{sz}(),
            aligned = Val{true}(),
            nontemporal = Val{_nt}(),
        )

        run(f, A, pipe, params, nt)
    end

    # Prepare for the random access benchmarks.
    for vector_size in params.vector_sizes
        sz = params.array_size

        # Get the number of unique indices to cycle through.
        lfsr_size = div(2 ^ sz, vector_size)
        for _nt in params.temporals
            nt = (
                vector = Val{vector_size}(),
                lfsr = LFSR(lfsr_size),
                nontemporal = Val{_nt}(),
            )

            run(hop_sum, A, pipe, params, nt)
            run(hop_write, A, pipe, params, nt)
            run(hop_increment, A, pipe, params, nt)
        end
    end
    return nothing
end

function run(f, A, pipe, params, nt)
    # warm up the function.
    threadme(f, A, nt...; prepare = true, iterations = 1)
    iters = params.inner_iterations

    # Run all the desired measurements
    first = true
    for m in params.measurements
        # Create a named tuple of the all the run parameters.
        run_params = (
            sz = div(sizeof(A), 10^9),
            mode = params.mode,
            threads = Threads.nthreads(),
            array = params.array_type,
        )

        run_params = merge(run_params, nt)
        final_params = make_params(f, run_params)

        # Always store data in the same spot I guess ...
        filepath = joinpath(DATADIR, "array_data.jls")

        # Transfer the measurements and parameters to the `server` process.
        transfer(pipe, m)
        paramtransfer(pipe, final_params)

        println(pipe, "filepath $filepath")
        println(pipe, "start")
        sleep(2)
        threadme(f, A, nt...; iterations = iters)
        sleep(2)
        println(pipe, "stop")
    end
    return nothing
end

####
#### Main
####

# Hack to get Julia threads to stop being moved by the linux scheduler
function assign_affinities(cpustart = 24)
    # Get Julia's PID
    pid = getpid()

    # Get the threads from Julia. The first thread belonds to Julia.
    # The next 8 threads (by defualt) belong to BLAS.
    # The rest of the threads are extra Julia threads.
    threads = split(chomp(read(pipeline(`ps -e -T`, `grep $pid`), String)), "\n")

    julia_thread_strings = vcat(threads[1], threads[10:end])
    @assert length(julia_thread_strings) == Threads.nthreads()

    # Parse out the thread PID
    _parse(x) = parse(Int, split(x)[2])
    julia_threads = _parse.(julia_thread_strings)

    # Start assigning affinity.
    for (index, pid) in enumerate(julia_threads)
        Base.run(`taskset -cp $(cpustart - 1 + index) $pid`)
    end
    return nothing
end

#####
##### Kernel for testing if the DRAM cache is inclusive.
#####

function cache_smasher(size_per_thread, f = (sz) -> Vector{Float32}(undef, sz))
    # Determine how big our array should be and instantiate it.
    sz = Threads.nthreads() * (2 ^ size_per_thread)
    A = f(sz)
    # Make sure the OS instantiates the whole array.
    threadme(vector_write, A, Val{16}())

    # Now, make sure everything in the cache is clean.
    threadme(vector_sum, A, Val{16}())
    return _cache_smasher(A, f)
end

# Allow an entry point for if the array is already allocated.
function _cache_smasher(A, f)
    # Allocate another array that figs in the L3 cache (33 MB)
    #
    # Here, we make an array that is 4 MB.
    B = f(2^17)
    @assert eltype(B) == Float32

    # Pass in another vector to record timings.
    timings = Float64[]
    threadme(_smasher, A, B, timings)
    return timings
end

function _smasher(x, small, timings)
    # We want one thread to iterate over the small array.
    #
    # All other threads blow out the cache
    if Threads.threadid() == 1
        _cycle(small, timings)
    else
        # Sleep so the cycler can get some inital data
        sleep(5)
        for _ in 1:2
            vector_sum(x, Val{16}())
        end
    end
end

function _cycle(x, timings)
    # Time for 20 seconds.
    start = now()
    while now() < start + Second(60)
        t = @elapsed for _ in 1:1000
            vector_sum(x, Val{16}())
        end
        push!(timings, t)
    end
end
