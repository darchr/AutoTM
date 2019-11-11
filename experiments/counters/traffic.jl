# Instructions for use:
#
# Start julia with
#     sudo ~/julia-1.2.0/bin/julia
#
# -- We must not run under "sudo", but `counters.jl` must run under "sudo" in order to
# sample the counters.
#
# The script will run through a collection of AutoTM experiments to run, and before running
# will invoke a job on the second process to start sampling.
#
# The master process will write to a Named Pipe when the ngraph function has completed -
# at which point the secondary process will stop sampling and serialize its sampled data
# to a filepath provided by the master worker.


# The name of the Pipe to use for communication with `counter.jl`
pipe_name = "counter_pipe"

#####
##### Code Loading
#####

include("init.jl")
include("Traffic.jl")
using .Traffic: Traffic

using Dates
using ArgParse
using Sockets
using Serialization
using Random
using SIMD

# Like serialize, but will also make a directory if needed.
function save(file, x)
    dir = dirname(file)
    isdir(dir) || mkpath(dir)
    serialize(file, x)
    return nothing
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--array-size"
            required = true
            arg_type = Int
            help = "Allocate 2^[array-size] elements per thread."

        "--sampletime"
            arg_type = Int
            default = 1
            help = "The number of seconds between sampling the hardware counters"

        "--counter_region"
            arg_type = String
            default = "uncore"
            range_tester = x -> in(x, ("uncore", "core"))
            help = """
            Select between [core] and [uncore]. Default: uncore.
            """


        "--counter_type"
            arg_type = String
            default = ["rw"]
            nargs = '*'
            # Make sure argument is one of the tag groups we know about.
            range_tester = x -> in(x, ("rw", "tags", "queues", "dram-queues", "insert-check"))
            help = """
            Select which sets of counters to use.
            If argument is not provided, counters will not be used.
            [rw]: Use DRAM/PMEM read/write counters
            [tags]: Return dirty/clean miss count and hit count.
            [queues]: Return the read and write queue occupancy for PMM.
            [dram-queues]: Return the read and write queue occupancey for DRAM.
            [insert-check]: Check semantic difference of read/write insert vs read/write
            command counters.
            """

        "--mode"
            arg_type = String
            default = "2lm"
            range_tester = x -> in(x, ("1lm", "2lm"))
            help = """
            Tell what mode is being run: [1lm] or [2lm]. Default: 2lm
            """

        "--inner_iterations"
            arg_type = Int
            default = 10
            range_tester = x -> x > zero(x)
            help = """
            Number of times to repeat the inner function.
            """
    end

    return parse_args(s)
end

function benchmark(A::Array, parsed_args, pipe)
    # Set up a bunch of nested loops to run through all the variations of tests we want
    # to execute.
    fns = [
        # Run `sum` twice to make sure everything is compiled.
        Traffic.vector_sum,
        Traffic.vector_sum,
        Traffic.vector_write,
        Traffic.vector_increment
    ]
    vector_sizes = [16, 8]
    nontemporal = [false, true]

    for (f, sz, nt) in Iterators.product(fns, vector_sizes, nontemporal)
        # Create a NamedTuple of parameters to forward to the function.
        nt = (
            vector = Val{sz}(),
            aligned = Val{true}(),
            nontemporal = Val{nt}(),
        )

        #run(f, A, parsed_args, pipe, nt)
    end

    # `hop` is special.
    #run(Traffic.hop, A, parsed_args, pipe, NamedTuple())
    sz = parsed_args["array-size"]
    vector_size = 16
    lfsr_size = sz - convert(Int, log2(vector_size))

    nt = (
        vector = Val{vector_size}(),
        lfsr = Traffic.LFSR{lfsr_size}(Traffic.FEEDBACK[lfsr_size]),
    )

    run(Traffic.hop_sum, A, parsed_args, pipe, nt)
    run(Traffic.hop_sum, A, parsed_args, pipe, nt)
    run(Traffic.hop_write, A, parsed_args, pipe, nt)
    run(Traffic.hop_increment, A, parsed_args, pipe, nt)
    return nothing
end

function run(f, A, parsed_args, pipe, nt)
    # warm up the function.
    threadme(f, A, nt...; prepare = true)
    iters = parsed_args["inner_iterations"]

    counter_region = parsed_args["counter_region"]

    if counter_region == "uncore"
        for counter_set in parsed_args["counter_type"]
            # Create a named tuple of the all the run parameters.
            params = (
                sz = div(sizeof(A), 10^9),
                mode = parsed_args["mode"],
                counters = counter_set,
                threads = Threads.nthreads(),
            )

            params = merge(params, nt)
            filepath = Traffic.make_filename(f, params)
            @show filepath

            println("Running Counters: $(counter_set)")

            # Setup the counters
            println(pipe, "filepath $filepath")
            println(pipe, "counter_region uncore")
            println(pipe, "counters $(counter_set)")
            println(pipe, "start")
            sleep(2)
            threadme(f, A, nt...; iterations = iters)
            sleep(2)
            println(pipe, "stop")
        end
    else
        params = (
            sz = div(sizeof(A), 10^9),
            mode = parsed_args["mode"],
            counters = "core",
            threads = Threads.nthreads(),
        )

        params = merge(params, nt)
        filepath = Traffic.make_filename(f, params)
        @show filepath

        # Setup the counters
        println(pipe, "filepath $filepath")
        println(pipe, "counter_region core")
        println(pipe, "start")
        sleep(2)
        threadme(f, A, nt...; iterations = iters)
        sleep(2)
        println(pipe, "stop")
    end
    return nothing
end

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

function main()
    #assign_affinities()
    parsed_args = parse_commandline()

    # Allocate and instantiate the array
    A = Vector{Float32}(undef, Threads.nthreads() * (2 ^ parsed_args["array-size"]))
    threadme(Traffic.vector_write, A, Val{16}())

    # Setup the counter
    pipe = connect(pipe_name)
    println(pipe, "sampletime $(parsed_args["sampletime"])")

    # Do the benchmarking!
    benchmark(A, parsed_args, pipe)
end

# Invoke the `main` function
main()
