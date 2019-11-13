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
end

function runkernels(params, x::Type{T} = Vector{Float32}; delete_old = false) where T
    # Allocate and instantiate the array
    sz = Threads.nthreads() * (2 ^ params.array_size)
    @show sz
    A = T(undef, sz)
    threadme(vector_write, A, Val{16}())

    # Setup the counter
    pipe = connect(PIPEPATH)
    println(pipe, "sampletime $(params.sampletime)")

    # Do the benchmarking!
    benchmark(A, pipe, params; delete_old = delete_old)
end

function benchmark(A::AbstractArray, pipe, params::KernelParams; delete_old = false)
    # Set up a bunch of nested loops to run through all the variations of tests we want
    # to execute.
    fns = [
        # Run `sum` twice to make sure everything is compiled.
        vector_sum,
        vector_write,
        vector_increment
    ]
    vector_sizes = [16]
    nontemporal = [false, true]

    for (f, sz, _nt) in Iterators.product(fns, vector_sizes, nontemporal)
        # Create a NamedTuple of parameters to forward to the function.
        nt = (
            vector = Val{sz}(),
            aligned = Val{true}(),
            nontemporal = Val{_nt}(),
        )

        #run(f, A, pipe, params, nt; delete_old = delete_old)
    end

    # Prepare for the random access benchmarks.
    sz = params.array_size
    vector_size = 16
    lfsr_size = sz - convert(Int, log2(vector_size))

    for _nt in nontemporal
        nt = (
            vector = Val{vector_size}(),
            lfsr = LFSR{lfsr_size}(FEEDBACK[lfsr_size]),
            nontemporal = Val{_nt}(),
        )

        run(hop_sum, A, pipe, params, nt; delete_old = delete_old)
        run(hop_write, A, pipe, params, nt; delete_old = delete_old)
        run(hop_increment, A, pipe, params, nt; delete_old = delete_old)
    end
    return nothing
end

function run(f, A, pipe, params, nt; delete_old = false)
    # warm up the function.
    threadme(f, A, nt...; prepare = true, iterations = 1)
    iters = params.inner_iterations

    # Run all the desired measurements
    first = true
    for m in params.measurements
        # Create a named tuple of the all the run parameters.
        params = (
            sz = div(sizeof(A), 10^9),
            mode = params.mode,
            threads = Threads.nthreads(),
        )

        params = merge(params, nt)
        filepath = make_filename(f, params)

        if first && delete_old
            ispath(filepath) && rm(filepath)
            first = false
        end

        # Transfer the measurements to the `server` process.
        transfer(pipe, m)

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

