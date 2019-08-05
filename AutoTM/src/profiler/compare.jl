# Timing methods for the whole function
function gettime(fex::nGraph.FluxExecutable; timeout = Second(10), min_calls = 3)
    start = now()
    mintime = typemax(Float64)
    times = 1
    while (now() < start + timeout) || (times <= min_calls)
        @info "Running Function"
        runtime = @elapsed(read(fex()))
        @info "Done Running Function"
        mintime = min(mintime, runtime)
        times += 1
    end
    return mintime
end

function get_baseline_allocation(backend, f)
    allocation_ref = Ref{Int}(0)
    io_ref = Ref{Int}(0)
    callbacks = CallbackChain()

    # For the first callback, we want to set all intermediate algorithms to their fastest
    # setting.
    function set(f::nGraph.NFunction)
        data = profile(f, backend)
        for node in nodes(data)
            if nGraph.Lib.can_select_algo(nGraph.getpointer(node))
                # Find the minimum runtime of the algorithms
                runtimes = get_times(gettime(data, node))
                _, ind = findmin(runtimes)
                enum = get_enums(gettime(data, node))[ind]

                nGraph.Lib.set_algo(
                    nGraph.getpointer(node),
                    convert(UInt, enum),
                    convert(UInt, get_bytes(gettime(data, node), enum)),
                )
            end
        end
    end

    function ret(f::nGraph.NFunction)
        allocation_ref[] = nGraph.get_temporary_pool_size(f)
        io_ref[] = sum(sizeof, input_tensors(f)) + sum(sizeof, output_tensors(f)) 
        throw(CompilerExit())
    end

    callback!(callbacks, set)
    callback!(callbacks, ret)

    try
        return actualize(backend, f; callback = callbacks)
    catch e
        isa(e, CompilerExit) || rethrow(e)
    end

    return allocation_ref[], io_ref[]
end


_base_stats() = (
    io_size = Ref(0),
    default_alloc_size = Ref(0),
    # GPU runtime usinc CUDA Malloc Managed
    gpu_managed_runtime = Ref(0.0),
    runs = Vector{Dict{Symbol,Any}}(),
)

"""
    compare(f, opt_iter; kw...)

* `f`: A constructor function `f() -> fex, args` returning a `FluxExecutable` and a tuple
    of arguments to be passed to the executable.

* `opt_iter`: An iterator returning optimization arguments that will be passed to `factory`.

Keywords
--------

* `cache`: A cache for kernel timings. Defaults to `CPUKernelCache(BASE_CACHE_PATH)`. That
    is, the default cache.

* `statspath`: An optional file path to saved stats. If this is given, any DRAM limits
    already in the cached stats will be skipped on this profiling run.
"""
function compare(
        func,
        opt,
        backend::nGraph.Backend;
        skip_base_check = false,
        statspath = nothing,
        kw...
    )

    if (isnothing(statspath) || !ispath(statspath))
        stats = _base_stats()
        if !skip_base_check
            initialize!(stats, func, backend)
        end
    else
        stats = deserialize(statspath)
    end

    # TODO: This inner call is left over from legacy code and doesn't really make sense
    # anymore.
    #
    # A future refactoring could bring this call up.
    GC.gc()
    _compare!(
        stats,
        func,
        opt,
        backend;
        kw...
    )

    @info """
    Predicted Run Time: $(last(stats.runs)[:predicted_runtime])
    Actual Run Time: $(last(stats.runs)[:actual_runtime])
    """

    sort!(stats.runs; rev = true, by = x -> x[:dram_limit])
    isnothing(statspath) || serialize(statspath, stats)

    return stats
end

function initialize!(stats, func, backend::nGraph.Backend{nGraph.CPU})
    # Instantiate the function
    fex = actualize(backend, func)

    stats.io_size[] = sum(sizeof, input_tensors(fex)) + sum(sizeof, output_tensors(fex))
    stats.default_alloc_size[] = nGraph.get_temporary_pool_size(fex.ex.ngraph_function)
    return nothing
end

# To initialize the GPU stuff, we turn on the Managed Memory flag and compile the function
function initialize!(stats, func, backend::nGraph.Backend{nGraph.GPU})
    fex = withenv("NGRAPH_GPU_CUDA_MALLOC_MANAGED" => true) do 
        actualize(backend, func)
    end

    stats.io_size[] = sum(sizeof, input_tensors(fex)) + sum(sizeof, output_tensors(fex))
    stats.default_alloc_size[] = nGraph.get_temporary_pool_size(fex.ex.ngraph_function)
    stats.gpu_managed_runtime[] = gettime(fex)
end

function _compare!(stats, f, opt, backend::nGraph.Backend{nGraph.CPU}; skip_run = false, kw...)
    fex, frame, _metadata = factory(backend, f, opt; kw...)
    GC.gc()
    data = frame.profile_data

    # Unpack some of the return values for logging
    creation_times = _metadata[:creation_times]
    optimization_times = _metadata[:optimization_times]

    # Get the predicted run time and then the actual run time
    nt = Dict(
        :creation_times => creation_times,
        :optimization_times => optimization_times,
        :predicted_runtime => Runner.predict(frame),
        :dram_limit => maxlimit(frame.modeltype),
        :tensor_size_map => Dict(nGraph.name(t) => sizeof(t) for t in tensors(data)),
        :config_map => Dict(nGraph.name(n) => getconfig(nGraph.Node(n)) for n in nodes(data)),
    )

    nt_new = Dict(
        # Some statistics on nodes and tensors

        # Number of move nodes plus bytes moved around
        :num_move_nodes => count(_move_filter(), nodes(data)),
        :num_pmem_move_nodes => count(_move_filter(PMEM), nodes(data)),
        :num_dram_move_nodes => count(_move_filter(DRAM), nodes(data)),

        :bytes_moved => _count(inputs, sizeof, data; filt = _move_filter()),
        :bytes_moved_pmem => _count(inputs, sizeof, data; filt = _move_filter(PMEM)),
        :bytes_moved_dram => _count(inputs, sizeof, data; filt = _move_filter(DRAM)),

        :num_async_move_nodes => count(_async_filter(), nodes(data)),
        :num_pmem_async_move_nodes => count(_async_filter(PMEM), nodes(data)),
        :num_dram_async_move_nodes => count(_async_filter(DRAM), nodes(data)),

        :bytes_async_moved => _count(inputs, sizeof, data; filt = _async_filter()),
        :bytes_async_moved_pmem => _count(inputs, sizeof, data; filt = _async_filter(PMEM)),
        :bytes_async_moved_dram => _count(inputs, sizeof, data; filt = _async_filter(DRAM)),

        # Total number of kernels
        :num_kernels => count(hasprofile, nodes(data)),
        :num_input_tensors => _count(inputs, data; filt = hasprofile),
        :num_output_tensors => _count(outputs, data; filt = hasprofile),

        :num_dram_input_tensors => _count(
            x -> filter(!nGraph.is_persistent, inputs(x)),
            data; filt = hasprofile
        ),
        :num_dram_output_tensors => _count(
            x -> filter(!nGraph.is_persistent, outputs(x)),
            data; filt = hasprofile
        ),

        # Get the sizes of the input and output tensors
        :bytes_input_tensors => _count(inputs, sizeof, data; filt = hasprofile),
        :bytes_output_tensors => _count(outputs, sizeof, data; filt = hasprofile),

        :bytes_dram_input_tensors => _count(
            x -> filter(!nGraph.is_persistent, inputs(x)),
            sizeof,
            data;
            filt = hasprofile
        ),
        :bytes_dram_output_tensors => _count(
            x -> filter(!nGraph.is_persistent, outputs(x)),
            sizeof,
            data;
            filt = hasprofile
        ),

        # Info on global allocations
        :dram_alloc_size => nGraph.get_temporary_pool_size(fex.ex.ngraph_function),
        :pmem_alloc_size => nGraph.get_pmem_pool_size(fex.ex.ngraph_function),
        :move_time => estimate_move_time(fex, frame),
    )
    nt = merge(nt, nt_new)

    if !skip_run
        nt_new = Dict(
            :actual_runtime => gettime(fex),
            :kernel_times => read_timing_data(fex.ex.ngraph_function)
        )
        nt = merge(nt, nt_new)
    end

    push!(stats.runs, nt)

    return nothing
end

function _compare!(stats, f, opt, backend::nGraph.Backend{nGraph.GPU}; kw...)
    fex, frame = factory(backend, f, opt; kw...)
    GC.gc()
    data = frame.profile_data

    # Get the predicted run time and then the actual run time
    nt = Dict(
        :predicted_runtime => Runner.predict(frame),
        :dram_limit => maxlimit(frame.modeltype),
        :tensor_size_map => Dict(nGraph.name(t) => sizeof(t) for t in tensors(data)),

        # Number of move nodes plus bytes moved around
        :num_move_nodes => count(_move_filter(), nodes(data)),
        :num_pmem_move_nodes => count(_move_filter(PMEM), nodes(data)),
        :num_dram_move_nodes => count(_move_filter(DRAM), nodes(data)),

        :bytes_moved => _count(inputs, sizeof, data; filt = _move_filter()),
        :bytes_moved_pmem => _count(inputs, sizeof, data; filt = _move_filter(PMEM)),
        :bytes_moved_dram => _count(inputs, sizeof, data; filt = _move_filter(DRAM)),

        :num_async_move_nodes => count(_async_filter(), nodes(data)),
        :num_pmem_async_move_nodes => count(_async_filter(PMEM), nodes(data)),
        :num_dram_async_move_nodes => count(_async_filter(DRAM), nodes(data)),

        :bytes_async_moved => _count(inputs, sizeof, data; filt = _async_filter()),
        :bytes_async_moved_pmem => _count(inputs, sizeof, data; filt = _async_filter(PMEM)),
        :bytes_async_moved_dram => _count(inputs, sizeof, data; filt = _async_filter(DRAM)),

        # Total number of kernels
        :num_kernels => count(hasprofile, nodes(data)),
        :num_input_tensors => _count(inputs, data; filt = hasprofile),
        :num_output_tensors => _count(outputs, data; filt = hasprofile),

        :num_dram_input_tensors => _count(
            x -> filter(!nGraph.is_persistent, inputs(x)),
            data; filt = hasprofile
        ),
        :num_dram_output_tensors => _count(
            x -> filter(!nGraph.is_persistent, outputs(x)),
            data; filt = hasprofile
        ),

        # Get the sizes of the input and output tensors
        :bytes_input_tensors => _count(inputs, sizeof, data; filt = hasprofile),
        :bytes_output_tensors => _count(outputs, sizeof, data; filt = hasprofile),

        :bytes_dram_input_tensors => _count(
            x -> filter(!nGraph.is_persistent, inputs(x)),
            sizeof,
            data;
            filt = hasprofile
        ),
        :bytes_dram_output_tensors => _count(
            x -> filter(!nGraph.is_persistent, outputs(x)),
            sizeof,
            data;
            filt = hasprofile
        ),

        # Info on global allocations
        :dram_alloc_size => nGraph.get_temporary_pool_size(fex.ex.ngraph_function),
        :pmem_alloc_size => nGraph.get_pmem_pool_size(fex.ex.ngraph_function),
        :move_time => estimate_move_time(fex, frame),

        # Timing Breakdowns
        :actual_runtime => gettime(fex),
        :oracle_time => fastest_time(frame),
        #:kernel_times => read_timing_data(fex.ex.ngraph_function)
    )

    push!(stats.runs, nt)
    return nothing
end

#####
##### Intraction methods with the `rettuple` from `compare`
#####

dc(x) = all(isequal(Runner.DRAM), x)
pmem_count(x) = count(isequal(Runner.PMEM), x)

function gettimings(data)
    timings = NamedTuple[]

    for node in data.nodes
        hasprofile(node) || continue
        configs = collect(keys(node.timings))

        dram_config = configs[findfirst(x -> dc(x.inputs) && dc(x.outputs), configs)]
        input_dram = filter(x -> dc(x.inputs), configs) |> collect
        output_dram = filter(x -> dc(x.outputs), configs) |> collect

        # Find the configs with the most inputs in PMEM with all outputs in DRAM
        # and find the config with the most outputs in PMEM with all inputs in DRAM
        _, i = findmax(map(x -> pmem_count(x.inputs), output_dram))
        max_input_pmem_config = output_dram[i]

        _, i = findmax(map(x -> pmem_count(x.outputs), input_dram))
        max_output_pmem_config = input_dram[i]

        # Find the comfig with the most number of PMEM io
        _, i = findmax(map(x -> pmem_count(x.inputs) + pmem_count(x.outputs), configs))
        max_pmem_config = configs[i]

        nt = (
            description = node.description,
            dram = minimum(node.timings[dram_config]),
            pmem = minimum(node.timings[max_pmem_config]),
            input_pmem = minimum(node.timings[max_input_pmem_config]),
            output_pmem = minimum(node.timings[max_output_pmem_config]),
        )
        push!(timings, nt)
    end
    return timings
end

#####
##### Compare the running times of a function with the predicted runtime.
#####

function compare_kernel_times(fex::nGraph.FluxExecutable, data::ProfileData)
    kernel_times = read_timing_data(fex.ex.ngraph_function)
    results = []

    # Iterate through the kernels - find kernels with timing parameter, get their time,
    # and then find what the expected runtime is.
    for op in fex.ex.ngraph_function
        op_wrapped = NodeDescriptor(op)
        if !hasprofile(op_wrapped) || description(op_wrapped) == "Move"
            continue
        end

        op_name = name(op_wrapped)
        config = getconfig(op)

        # Get the actual run time.
        index = findfirst(x -> x["name"] == op_name, kernel_times)
        actual_runtime = kernel_times[index]["dur"]

        # Get the expected run time
        index = findfirst(isequal(op_wrapped), nodes(data))
        expected_time = gettime(data, nodes(data, index), config)

        push!(results, (
            name = op_name,
            config = config,
            actual = actual_runtime,
            expected = expected_time,
            node = op_wrapped,
        ))
    end
    return results
end

