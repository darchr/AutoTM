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
        if nGraph.Lib.can_select_algo(nGraph.getpointer(unx(node)))
            algo_var = frame.model[:algo_var]
            local algo_enum
            for enum in enums(gettime(node))
                if approx_one(algo_var[node, enum])
                    algo_enum = enum
                    break
                end
            end

            actual = perf[nGraph.name(node)]
            expected = timeat(gettime(node), algo_enum)

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
            expected = gettime(node, getconfig(nGraph.Node(unx(node))))

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
        if nGraph.Lib.can_select_algo(nGraph.getpointer(unx(node)))
            time += minimum(times(gettime(node)))
        else
            time += gettime(node)
        end
    end

    return time
end

function fastest_time(fex, frame)
    # Take into account actual runtime.
    data = frame.profile_data
    kernel_times = nGraph.get_performance(fex.ex)

    time = 0.0
    for node in filter(hasprofile, nodes(data))
        if nGraph.Lib.can_select_algo(nGraph.getpointer(unx(node)))
            time += min(
                minimum(times(gettime(node))),
                kernel_times[nGraph.name(node)]
            )
        else
            time += min(
                gettime(node),
                kernel_times[nGraph.name(node)]
            )
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
