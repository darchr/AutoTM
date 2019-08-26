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

function profile(f::nGraph.NFunction, backend::nGraph.Backend{nGraph.CPU};
        cache = nothing,
        # Force re-profiling
        recache = false,
        kernel_tiling = 1,
    )

    isnothing(cache) && error("Need to define a cache")

    # Go through each node
    # Create new parameters with the same input and outputs sizes
    # Call `copy_with_new_args` on the node in question with the new parameters
    # Swip the input and output tensor layouts from the node in question
    data = FunctionData(f, nGraph.CPU)

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
            kernel_params = CPUKernelParams(unx(node))
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
        kernel_params = CPUKernelParams(unx(node))
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
        ex, inputs, outputs, copied_ops = extract(
            nGraph.Node(node),
            backend;
            ncopies = kernel_tiling,
        )

        # Profile the timings
        for config in filter(!in(cached_configs), configs)
            _update!(progress_bar, node, config, ncached)
            # setup the config
            for op in copied_ops
                _setup!(op, config)
            end

            # recompile the function to reflect the new config state
            GC.gc()
            ex = nGraph.recompile(ex)
            GC.gc()
            function_name = nGraph.name(ex.ngraph_function)

            # Run the inner loop multiple times to warm up the cache.
            # This seems to make a pretty big difference for smaller batchsizes.
            for _ in 1:3
                ex(inputs, outputs)
            end
            record_time!(
                data,
                node,
                function_name,
                copied_ops,
                config
            )

            _cleanup!(ex.ngraph_function)

            # Save the results to the cache, and then save the cache
            cache[(kernel_params, config)] = gettime(data, node, config)
            save(cache)
        end
    end

    # Renable optimization passes
    enable_passes()

    return data
end

read_timing_data(fn::nGraph.NFunction) = read_timing_data(nGraph.name(fn))
read_timing_data(fn::AbstractString) = JSON.parsefile("$fn.timeline.json")["traceEvents"]

function record_time!(
        data::FunctionData{nGraph.CPU},
        node::NodeDescriptor,
        function_name,
        ops::Vector{<:nGraph.Node},
        expected_config
    )

    timings = read_timing_data(function_name)
    # Get the persistence config of this op
    configs = getconfig.(ops)
    inds = findall(isequal(expected_config), configs)

    found_ops = ops[inds]

    # Make sure that the expected configuration is the one we actualy get.
    if isempty(found_ops)
        @error """
        Expected config $expected_config.
        Got Config $config.

        Op Names: $(nGraph.name.(ops))
        paramss: $(CPUKernelParams.(ops))
        """
        error()
    end

    # Extract the timings and record it
    times = Float64[]
    for op in found_ops
        index = findfirst(x -> x["name"] == nGraph.name(op), timings)
        @assert index !== nothing
        push!(times, timings[index]["dur"])
    end

    if hastime(data, node, expected_config)
        settime!(
            data,
            node,
            expected_config,
            min(gettime(data, node, expected_config), minimum(times))
        )
    else
        settime!(
            data,
            node,
            expected_config,
            minimum(times)
        )
    end
end

"""
    extract(node::nGraph.Node, backend::nGraph.Backend{nGraph.CPU}; ncopies = 1)

Create a `nGraph.Executable` with a copy of `node` using the same input and output
parameters. `Move` nodes will be inserted at all inputs and outputs of the copied node
if the inputs come from executable arguments and if the outputs go directly to results.

To try to capture some caching information, the node will be replicated `ncopies` times
in the returned executable.
"""
function extract(node::nGraph.Node, backend::nGraph.Backend{nGraph.CPU}; ncopies = 1)
    # Create parameters for the inputs
    params = nGraph.Node[]
    for i in 1:nGraph.get_input_size(node)
        A = rand(nGraph.get_input_element_type(node, i), nGraph.get_input_shape(node, i)...)
        push!(params, nGraph.parameter(A))
    end

    # Insert layout conversion to match the mkldnn layouts in the original graph.
    links = nGraph.Node[]
    for i in 1:nGraph.get_input_size(node)
        if nGraph.input_needs_conversion(node, i)
            push!(links, nGraph.convert_layout_to(params[i], node, i))
        else
            push!(links, params[i])
        end
    end

    # Copy the node with the newly created parameters
    copied_nodes = [copy(node, links) for _ in 1:ncopies]

    # Make sure we're using the same version of the node.
    nGraph.is_mkldnn(node) && nGraph.set_mkldnn.(copied_nodes)

    # Compile the new function
    paramvector = nGraph.ParameterVector(params...)

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

    # First, we compile the function
    ex = nGraph.compile(backend, paramvector, nodevector)

    # Then, we have to inspect the graph, find the nodes that do not have converted
    # inputs, and insert out synchronous move nodes so we can control the input/output
    # state of the node under test.
    #
    # But first, we have to find what happened to the original node and find it in the
    # new graph.
    translated_nodes = nGraph.Node[]
    for op in ex.ngraph_function
        # Line it up by description and input/output sizes.
        if CPUKernelParams(op) == CPUKernelParams(node)
            push!(translated_nodes, op)

        # Handle Special Cases
        elseif nGraph.description(op) == "MatmulBias" && nGraph.description(node) == "Dot"
            push!(translated_nodes, op)
        end
    end

    if isempty(translated_nodes)
        for n in copied_nodes
            println("Copied Node: $n")
            println("    name: $(nGraph.name(n))")
        end
        error("Something done gone wrong!")
    end

    for translated_node in translated_nodes
        for (index, input) in enumerate(nGraph.get_inputs(translated_node))
            if nGraph.description(input) == "Parameter"
                nGraph.splice(input, 1, translated_node, index, nGraph.move(input))
            end
        end
    end

    # Recompile the function, now with the move nodes.
    ex = nGraph.recompile(ex)

    # Make these any to make them compatible with the inner call for nGraph.Executable
    input_tensors = Any[nGraph.Tensor(backend, x).ptr for x in params]
    output_tensors = Any[nGraph.Tensor(backend, x).ptr for x in outputs]

    return ex, input_tensors, output_tensors, translated_nodes
end
