#####
##### _compare! - CPU Edition
#####
function Profiler._compare!(
        stats, 
        f, 
        opt::ILPOptimizer, 
        backend::nGraph.Backend{nGraph.CPU}; 
        skip_run = false, kw...
    )

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
        :predicted_runtime => predict(frame),
        :dram_limit => maxlimit(frame.modeltype),
        :tensor_size_map => Dict(nGraph.name(t) => sizeof(t) for t in tensors(data)),
        :config_map => Dict(nGraph.name(n) => getconfig(nGraph.Node(n)) for n in nodes(data)),
        :ratio => getratio(opt),
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

#####
##### _compare! - GPU Edition
#####
function _compare!(stats, f, opt::ILPOptimizer, backend::nGraph.Backend{nGraph.GPU}; kw...)
    fex, frame = factory(backend, f, opt; kw...)
    GC.gc()
    data = frame.profile_data

    # Get the predicted run time and then the actual run time
    nt = Dict(
        :predicted_runtime => predict(frame),
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
