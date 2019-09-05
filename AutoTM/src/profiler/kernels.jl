# Profile the running times of kernels.
#
# Steps that need to be performed
#
# 1. Get the ops in the executable. Clone each of the ops into their own nGraph
#   function.
#
# 2. For each of the ops, we also have to capture the input and output tensor layouts
#   for the op in order to get accurate timing.
#
#   As far as I can tell, if a CPU op is annotated as "use_mkldnn_kernel", then the
#   input and output layout will always be the same. Thus, to do the correct
#   input/output layout conversion, we just have to check if the node we are copying is
#   annotated as mkldnn and make sure we annotate the new node as such.

disable_passes() = ENV["NGRAPH_PASS_ENABLES"] = join((
    "AlgebraicSimplification:0",
    "CoreFusion:0",
    "CPUFusion:0",
    "CPUHorizontalFusion:0",
    "CommonSubexpressionElimination:0",
    "ReshapeElimination:0",
   ), ";"
)

enable_passes() = delete!(ENV, "NGRAPH_PASS_ENABLES")

"""
    profile(args...)

Profile all of the operations in `fex`.

Keyword Arguments
-----------------
* `cache`: A cache to serve running times if a kernel has already been profiled. The cache
    must implement the function `save`.
"""
profile(fex::nGraph.FluxExecutable; kw...) =
    profile(fex.ex.ngraph_function, fex.ex.backend; kw...)

function profile(
        f::nGraph.NFunction, 
        backend::nGraph.Backend{T};
        cache = nothing,
        # Force re-profiling
        recache = false,
        kernel_tiling = 1,
    ) where {T}

    isnothing(cache) && error("Need to define a cache")

    # Create a "FunctionData" object for this function, which allows us to record the graph
    # structures and metadata on the Julia level.
    data = FunctionData(f, T)
    handle_algo_selection!(cache, backend, data)

    # Get all the configurations we are interested in for this run.
    # Need to make a MOVE node in order to control IO configurations.
    all_configs = possible_configs(data)

    # Convert the configs to a dictionary mapping node name to configs for easier
    # management
    config_dict = Dict{XNode, Vector{IOConfig}}()
    for config in all_configs
        v = get!(config_dict, first(config), IOConfig[])
        push!(v, last(config))
    end

    # Clean up cached configs if passed the `recache` option
    if recache
        @info "Removing Cached Configs"
        for node in nodes(data)
            hasprofile(node) || continue
            configs = config_dict[node]
            kernel_params = params(backend, node)
            for config in configs
                key = (kernel_params, config)
                delete!(cache, key)
            end
        end
    end

    num_configs = sum(length(config_dict[node]) for node in nodes(data) if hasprofile(node))
    progress_bar = Progress(num_configs, 1)

    # Setup a little update function for all configurations
    # This gives a more fine-grained information than updating for each op
    serviced = Ref(0)
    function _update!(p, op, config, ncached)
        serviced[] += 1
        ProgressMeter.next!(
            p;
            valuecolor = :white,
            showvalues = [
                (:iter, serviced[]),
                (:total, num_configs),
                (:op, nGraph.name(op)),
                (:config, config),
                (:ncached, ncached),
            ]
        )
    end

    # Disable some passes that get rid of the node we actually want
    disable_passes()

    ncached = 0
    for (index, node) in enumerate(nodes(data))
        # Skip unneeded ops
        hasprofile(node) || continue

        # Get the configs to run for this node
        configs = config_dict[node]

        # Before we build a sub-function, get all of the cached ops.
        cached_configs = IOConfig[]
        kernel_params = params(backend, node)
        for config in configs
            key = (kernel_params, config)
            if haskey(cache, key)
                # Update the number of timings serviced from cached ops
                ncached += 1
                _update!(progress_bar, node, config, ncached)

                settime!(node, config, cache[key])
                push!(cached_configs, config)
            end
        end

        # Abort if everything is cached
        length(cached_configs) == length(configs) && continue

        # Extract a subgraph with just this op
        for config in Iterators.filter(!in(cached_configs), configs)
            _update!(progress_bar, node, config, ncached)

            # Extract this particular node with the given configuration
            # Place GC barriers around the instantiation of the nGraph.Executable to try
            # to reduce interference with other extractions.
            GC.gc()
            ex, inputs, outputs, copied_ops = extract(
                nGraph.Node(unx(node)),
                backend,
                config;
                ncopies = kernel_tiling,
            )
            GC.gc()

            # Run the function several times. The first time is a warm up.
            # Timing for the second runs should be pretty consistent.
            for _ in 1:5
                ex(inputs, outputs)
            end

            # Save the run time into the node
            record_time!(node, ex, copied_ops, config)

            # Record the saved time into the cache and save the cache to persist the data
            # across transient or intentional (i.e. ctrl+c) failures.
            cache[(kernel_params, config)] = gettime(node, config)
            save(cache)
        end
    end

    # Renable optimization passes
    enable_passes()

    return data
end

#####
##### Algorithm selection
#####

# Right now, no algorithm selection happens in the CPU case - so just special case this 
# function to save some CPU cycles
handle_algo_selection!(cache, backend::nGraph.Backend{nGraph.CPU}, data::FunctionData) = nothing
function handle_algo_selection!(cache, backend::nGraph.Backend{nGraph.GPU}, data::FunctionData)
    for (index, node) in enumerate(nodes(data))
        hasprofile(node) || continue

        # Get the lookup key for this node
        kernel_params = params(backend, node)

        # If we can select an algorithm for this node, use the built-in timings
        # and skip the actual profiling.
        if nGraph.Lib.can_select_algo(nGraph.getpointer(unx(node))) 
            # Hack for the moment - make sure that we only have a single configuration
            # in the GPU case.
            #
            # If we need more flexibility - we'll deal with it in the future
            configs = possible_configs(node, nGraph.GPU)
            @assert length(configs) == 1
            config = first(configs)

            if !haskey(cache, (kernel_params, config))
                # Cleanup
                @info "Getting CUDNN timings for $(nGraph.name(node))"

                enums = UInt32[]
                times = Float32[]
                bytes = UInt64[]
                GC.gc()
                alloc_failed = nGraph.Lib.get_algo_options(
                    nGraph.getpointer(unx(node)),
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

                # NOTE: At one point, I was checking to make sure that we caught all 
                # allocation failures.
                #
                # However, it turned out that the majority of failures happened regardless 
                # of whether we were in a new session or not, so catching these failures
                # didn't really do much.
                #
                # I should probably remove this code.
                #
                # !allow_alloc_fail && alloc_failed && throw(error("""
                #     Not enough memory cleaned up to sufficiently profile everything.
                #     Restart the process and try again.
                #     """))

                cache[(kernel_params, config)] = algo_list
                save(cache)
            end
        end
    end
    return nothing
end

#####
##### Record the profiled runtime of a node
#####

function record_time!(
            node::XNode,
            ex::nGraph.Executable,
            ops::Vector{nGraph.Node},
            @nospecialize(config::IOConfig),
        )

    # First, make sure that all of the copied ops given for recording data actually have
    # the same IOConfig as what is expected
    configs = getconfig.(ops)
    inds = findall(isequal(config), configs)

    if isempty(inds)
        @show configs
        @show config
        error()
    end

    # Just pull out the ops that match this configuration
    ops = ops[inds]

    # Get a dictionary mapping node name to runtime in microseconds
    timing_dict = nGraph.get_performance(ex)

    # Get all of the runtimes for the ops being profiled
    times = [timing_dict[nGraph.name(op)] for op in ops]

    # Set the runtime for this node as the average of the runtimes of the individual
    # profiled ops.
    settime!(node, config, mean(times))
    return nothing
end

"""
    extract(node::nGraph.Node, backend::nGraph.Backend{nGraph.CPU}, config; ncopies = 1)

Create a `nGraph.Executable` with a copy of `node` using the same input and output
parameters. `Move` nodes will be inserted at all inputs and outputs of the copied node
if the inputs come from executable arguments and if the outputs go directly to results.

To try to capture some caching information, the node will be replicated `ncopies` times
in the returned executable.
"""
function extract(
        node::nGraph.Node,
        backend::nGraph.Backend,
        @nospecialize(config::IOConfig);
        ncopies = 1
    )
    # Create parameters for the inputs
    parameters = nGraph.Node[]
    for i in 1:nGraph.get_input_size(node)
        # Check if this input is a constant. If so - copy over the constant.
        this_input = nGraph.get_input(node, i)
        if isconstant(this_input)
            push!(parameters, nGraph.copy(this_input, nGraph.NodeVector()))
        else
            A = rand(nGraph.get_input_element_type(node, i), nGraph.get_input_shape(node, i)...)
            push!(parameters, nGraph.parameter(A))
        end
    end

    # Insert layout conversion to match the mkldnn layouts in the original graph.
    links = nGraph.Node[]
    for i in 1:nGraph.get_input_size(node)
        if nGraph.input_needs_conversion(node, i)
            push!(links, nGraph.convert_layout_to(parameters[i], node, i))
        else
            push!(links, parameters[i])
        end
    end

    # Copy the node with the newly created parameters
    copied_nodes = [copy(node, links) for _ in 1:ncopies]

    # Make sure we're using the same version of the node - will always return `false` if 
    # compiling for the GPU backend.
    nGraph.is_mkldnn(node) && nGraph.set_mkldnn.(copied_nodes)

    # Compile the new function
    #
    # Now that we've created a copy of the new node, we need to filter out all of the input
    # constants we provided so it compiles correctly.
    filter!(!isconstant, parameters) 
    paramvector = nGraph.ParameterVector(parameters...)

    outputs = nGraph.Node[]
    for copied_node in copied_nodes
        if nGraph.get_output_size(copied_node) > 1
            for i in 1:nGraph.get_output_size(copied_node)
                push!(outputs, nGraph.get_output_element(copied_node, i))
            end
        else
            push!(outputs, copied_node)
        end
    end

    # Get an result output for each output of the node
    nodevector = nGraph.NodeVector(outputs)

    # Create a pass callback to setup the function to the given configuration.
    translated_nodes_ref = Ref(nGraph.Node[])
    function configuration_callback(fn)
        # We have to inspect the graph, find the nodes that do not have converted
        # inputs, and insert out synchronous move nodes so we can control the input/output
        # state of the node under test.
        #
        # But first, we have to find what happened to the original node and find it in the
        # new graph.
        translated_nodes = nGraph.Node[]
        for op in fn
            # Line it up by description and input/output sizes.
            if params(backend, op) == params(backend, node)
                push!(translated_nodes, op)

            # Handle Special Cases
            # TODO: This should now be handled by turning off optimizations.
            # elseif nGraph.description(op) == "MatmulBias" && nGraph.description(node) == "Dot"
            #     push!(translated_nodes, op)
            end
        end

        # Throw an error if we didn't find a node matching what we expected.
        if isempty(translated_nodes)
            for n in copied_nodes
                println("Copied Node: $n")
                println("    name:   $(nGraph.name(n))")
                println("    config: $(config)")
            end
            error("Something done gone wrong!")
        end

        # Here, we splice Move nodes after any direct input to a translated_node that 
        # a Parameter.
        #
        # Originally, this to facilitate recompilation, but now that we're moving away from
        # recompilation, opting for adding a pass callback to the nGraph compilation 
        # pipeline, it may be time to revisit this.
        for translated_node in translated_nodes
            for (index, input) in enumerate(nGraph.get_inputs(translated_node))
                if nGraph.description(input) == "Parameter" && config[index] == PMEM
                    nGraph.splice(input, 1, translated_node, index, nGraph.move(input))
                end
            end
        end

        # Finally, configure the translated_nodes to the form that we want.
        for translated_node in translated_nodes 
            _setup!(translated_node, config)
        end

        # Save the translated nodes to the external translated_nodes_ref
        translated_nodes_ref[] = translated_nodes 
        return nothing
    end

    # Setup the above function as a callback
    callbacks = CallbackChain()
    callback!(callbacks, configuration_callback)

    # Compile the function
    ex = nGraph.compile(
        backend, 
        paramvector, 
        nodevector; 
        callback = callbacks, 
        emit_timing = true
    )

    # Make these any to make them compatible with the inner call for nGraph.Executable
    input_tensors = Any[nGraph.Tensor(backend, x).ptr for x in parameters]
    output_tensors = Any[nGraph.Tensor(backend, x).ptr for x in outputs]

    return ex, input_tensors, output_tensors, translated_nodes_ref[]
end
