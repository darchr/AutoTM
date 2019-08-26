# Some notes on the GPU implementation
# ------------------------------------
# While the CPU implmentation does its horrid hack of first compiling the graph, mutating
# it, and recompiling with most of the passes disabled, the GPU implmentation takes a
# different strategy.
#
# I've inserted a callback site in the GPU pass manager that lets us insert any `julia`
# function with no arguments as a callback.
#
# This callback is done ``before`` the function finishes compilation, so we don't really
# have to worry about function recompilation.
#
# The callback we insert is intended to
# - profile all the nodes in the function
# - determine which nodes have optional implementations and get the running time and
#       workspace size of these kernels. (This has a lot of added code on the C++ side
#       to facilitate this)
# - perform the graph optimization
#
# The reason I didn't do this for the CPU code is because didn't really think of it at the
# time. With some work, this strategy could be unified across CPU and GPu and would actually
# probably reduce a lot of the cluster that is the CPU code :(
function profile(f::nGraph.NFunction, backend::nGraph.Backend{nGraph.GPU};
        cache = nothing,
        allow_alloc_fail = false,
        recache = false,
        # Other keywords to make compatible with the CPU profile
        kw...
    )

    isnothing(cache) && error("Need to define a cache")

    data = FunctionData(f, nGraph.GPU)

    # Clean up cached configs if passed the `recache` option
    if recache
        @info "Removing Cached Configs"
        for node in nodes(data)
            hasprofile(node) || continue
            kernel_params = GPUKernelParams(node)
            delete!(cache, kernel_params)
        end
    end


    # We follow a strategy similar to the CPU, except much less fancy.
    #
    # Simply
    # - call `copy_with_new_args` on each node in question
    # - make a function with the correct inputs and outputs
    # - compile the function with no GPU callback
    #
    # NOTE: If `can_select_algo` for a node, we don't need to profile that node since
    # - All the `can_select_algo` nodes are CUDNN kernels
    # - CUDNN gives the expected performance of these kernels already :D

    # Work with all the `can_select_algo` nodes first.
    # Finalizers for small GPU structs don't clean up properly yet, so trying to profile
    # these workloads runs out of memory
    for (index, node) in enumerate(nodes(data))
        hasprofile(node) || continue

        # Get the lookup key for this node
        kernel_params = GPUKernelParams(node)

        # If we can select an algorithm for this node, use the built-in timings
        # and skip the actual profiling.
        if nGraph.Lib.can_select_algo(nGraph.getpointer(node))
            if !haskey(cache, kernel_params)
                # Cleanup
                @info "Getting CUDNN timings for $(nGraph.name(node))"

                enums = UInt32[]
                times = Float32[]
                bytes = UInt64[]
                GC.gc()
                alloc_failed = nGraph.Lib.get_algo_options(
                    nGraph.getpointer(node),
                    enums,
                    times,
                    bytes
                )

                # cuDNN returns the results of its time in milliseconds.
                #
                # Convert to microseconds here to make it uniform with the rest of 
                # the timings.
                algo_list = [
                    AlgorithmPerf(e, 1000 * t, b) for (e,t,b) in zip(enums, times, bytes)
                ]

                !allow_alloc_fail && alloc_failed && throw(error("""
                    Not enough memory cleaned up to sufficiently profile everything.
                    Restart the process and try again.
                    """))

                cache[kernel_params] = algo_list
                save(cache)
            end
        end
    end

    for (index, node) in enumerate(nodes(data))
        hasprofile(node) || continue

        # Get the lookup key for this node
        kernel_params = GPUKernelParams(node)

        # Get saved data from the cache if it exists
        if haskey(cache, kernel_params)
            settime!(data, node, cache[kernel_params])
            continue
        end

        # Skip profiling if already performed
        nGraph.Lib.can_select_algo(nGraph.getpointer(node)) && continue

        @info "Profiling $(nGraph.name(node))"

        # Extract the node in question and profile it
        ex, inputs, outputs, copied_op = extract(nGraph.Node(node), backend)

        # Run the function
        for _ in 1:10
            ex(inputs, outputs)
        end

        record_time!(data, node, ex, copied_op)
        GC.gc()
        cache[kernel_params] = gettime(data, node)
        save(cache)
    end

    return data
end

function record_time!(data::FunctionData{nGraph.GPU}, node, ex::nGraph.Executable, copied_op)
    timing_dict = nGraph.get_performance(ex)

    # Find the performance for the copied op in the timing dictionary
    time = timing_dict[nGraph.name(copied_op)]
    settime!(data, node, convert(Float64, time))
    return nothing
end

# GPU Node extraction
function extract(node::nGraph.Node, backend::nGraph.Backend{nGraph.GPU})
    params = nGraph.Node[]

    # Create parameters that are the same size as the input to this node
    for i in 1:nGraph.get_input_size(node)
        A = rand(nGraph.get_input_element_type(node, i), nGraph.get_input_shape(node, i)...)
        push!(params, nGraph.parameter(A))
    end

    # Copy the node with the newly created parameters
    copied_node = copy(node, params)
    paramvector = nGraph.ParameterVector(params...)

    # install "get_output_element" nodes
    outputs = nGraph.Node[]
    if nGraph.get_output_size(copied_node) > 1
        for i in 1:nGraph.get_output_size(copied_node)
            push!(outputs, nGraph.get_output_element(copied_node, i))
        end
    else
        push!(outputs, copied_node)
    end

    # Get an result output for each output of the node
    nodevector = nGraph.NodeVector(outputs)

    # First, we compile the function
    #
    # Make sure to emit timing.
    ex = nGraph.compile(backend, paramvector, nodevector; emit_timing = true)

    # Find the copied node in the new graph
    local translated_node
    found = false
    for op in ex.ngraph_function
        # Line it up by description and input/output sizes.
        if GPUKernelParams(op) == GPUKernelParams(copied_node)
            translated_node = op
            found = true
            break
        end
    end
    @assert found

    # Make these any to make them compatible with the inner call for nGraph.Executable
    input_tensors = Any[nGraph.Tensor(backend, x).ptr for x in params]
    output_tensors = Any[nGraph.Tensor(backend, x).ptr for x in outputs]

    return ex, input_tensors, output_tensors, translated_node
end

#####
##### Tools for checking profiling
#####

function check_profile(fex::nGraph.FluxExecutable, frame; only_greater = false)
    # Read the profile data from the function
    perf = nGraph.get_performance(fex.ex)

    data = frame.profile_data

    expected_total = 0.0
    actual_total = 0.0
    for node in nodes(data)
        hasprofile(node) || continue
        if nGraph.Lib.can_select_algo(nGraph.getpointer(node))
            algo_var = frame.model[:algo_var]
            local algo_enum
            for enum in get_enums(gettime(data, node))
                if approx_one(algo_var[node, enum])
                    algo_enum = enum
                    break
                end
            end

            actual = perf[nGraph.name(node)]
            expected = get_time(gettime(data, node), algo_enum)

            # Get the expected move time
            _async = get(frame.model[:tensor_async], nGraph.name(node), nothing)
            if !isnothing(_async)
                async_time = JuMP.value(_async)
            else
                async_time = 0.0
            end

            expected_total += expected
            actual_total += actual

            # Print out the results for this node.
            if !only_greater || actual > expected
                println("Algorithm selection for $(nGraph.name(node)): $algo_enum")
                println("    Actual Time: $(actual)")
                println("    Expected Time: $(expected)")
                println("    Async Time: $(async_time)")
                println()
            end
        else
            actual = perf[nGraph.name(node)]
            expected = gettime(data, node)

            # Get the expected move time
            _async = get(frame.model[:tensor_async], nGraph.name(node), nothing)
            if !isnothing(_async)
                async_time = JuMP.value(_async)
            else
                async_time = 0.0
            end

            expected_total += expected
            actual_total += actual

            if !only_greater || actual > expected
                println("No Algorithm selection for $(nGraph.name(node))")
                println("    Actual Time: $(actual)")
                println("    Expected Time: $(expected)")
                println("    Async Time: $(async_time)")
                println()
            end
        end
    end

    @info """
    Expected Total Time: $expected_total
    Actual Total Time: $actual_total
    """

    return nothing
end

function fastest_time(frame)
    data = frame.profile_data

    time = 0.0
    for node in filter(hasprofile, nodes(data))
        if nGraph.Lib.can_select_algo(nGraph.getpointer(node))
            time += minimum(get_times(gettime(data, node)))
        else
            time += gettime(data, node)
        end
    end

    return time
end

function show_algorithm_slowdown(frame)
    data = frame.profile_data
    model = frame.model

    for node in Iterators.filter(hasprofile, nodes(data))
        if nGraph.Lib.can_select_algo(nGraph.getpointer(node))
            printstyled("Checking node $(nGraph.name(node))\n"; color = :green)

            # Get the fastest executing algorithm
            time, ind = findmin(get_times(gettime(data, node)))
            enum = get_enums(gettime(data, node))[ind]
            println("    Fastest Enum: $enum. (time) $time")

            # Get the actual used algorithm
            algo_var = frame.model[:algo_var]
            local algo_enum
            for enum in get_enums(gettime(data, node))
                if approx_one(algo_var[node, enum])
                    algo_enum = enum
                    break
                end
            end

            time = get_time(gettime(data, node), algo_enum)
            println("    Actual Enum: $enum. (time) $(time)")
        end
    end
end
