# Build a workload, run it to get kernel times, and create a datastructure encoding the
# graph for later plotting runs.
function generate_timeline(backend, workload, cache, parsed_args)
    f = get_workload(workload)
    opt = get_optimizer(parsed_args["mode"])

    # Build up a list of optional keyword arguments.
    kw = []
    if parsed_args["use_2lm_scratchpad"]
        push!(kw, :use_scratchpad => true)
    end

    fex = AutoTM.Optimizer.factory(
        backend,
        f,
        opt;
        cache = cache,
        kw...
    ) |> maybeunwrap

    # Run twice - we'll get timing data from the second iteration.
    @time fex()

    # Reset the performance counters so they aren't "contaminated" from the original run.
    #
    # The performance counters accumulate across runs, so if we don't do this, we will
    # essentially be averaging the runtime from the second run with that of the first run.
    nGraph.reset_counters(fex.ex)
    @time fex()

    # Create a `FunctionData` object from the nGraph function.
    # Couple it with the performance metrics.
    function_data = AutoTM.Profiler.FunctionData(fex.ex.ngraph_function, backend)
    times = nGraph.get_performance(fex.ex)

    println("Sum of Kernel Run Times: $(sum(values(times)))")

    # Record metadata about each tensor that will be used for plot generation.
    tensor_records = map(collect(AutoTM.Profiler.tensors(function_data))) do tensor
        users = AutoTM.Profiler.users(tensor)
        return Dict(
             "name" => nGraph.name(tensor),
             "users" => nGraph.name.(users),
             "user_indices" => getproperty.(users, :index),
             "sizeof" => sizeof(tensor),
             "offset" => Int(AutoTM.Profiler.getoffset(tensor)),
        )
    end

    # Collect metadata on each node.
    node_records = map(AutoTM.Profiler.nodes(function_data)) do node
        outputs = nGraph.name.(AutoTM.Profiler.outputs(node))
        inputs = nGraph.name.(AutoTM.Profiler.inputs(node))
        time = get(times, nGraph.name(node), 0)
        delete!(times, nGraph.name(node))
        return Dict(
            "name" => nGraph.name(node),
            "inputs" => inputs,
            "outputs" => outputs,
            "time" => time,
        )
    end

    @assert isempty(times)

    # Combine together to the final record that will be saved.
    record = Dict(
        "tensors" => tensor_records,
        "nodes" => node_records,
    )

    file = make_filename(
        workload,
        parsed_args["mode"],
        "",
        parsed_args["use_2lm_scratchpad"];
        prefix = joinpath(@__DIR__, "serialized")
    )
    save(file, record)
    return nothing
end

# Tool for inspecting and generating the heap allocation of an executable.
#
# The modified nGraph allows us to pass two Julia callbacks to the compilation process.
# The first callback will be called before the memory allocation pass, and the second will
# be called AFTER memory allocatio pass.
#
# Since our graphs may exceed the physical memory of a system - we will just insert the
# extraction of the intermediate tensors from the second callback.
struct TensorRecord
    start::Float64
    stop::Float64
    offset::Float64
    height::Float64

    # Indices of reading and writing
    read_starts::Vector{Float64}
    read_stops::Vector{Float64}
    write_stop::Float64
end

function Base.isless(a::TensorRecord, b::TensorRecord)
    # First - order by starting index.
    if a.start < b.start
        return true

    # Then, order by offset
    elseif a.start == b.start
        if a.offset < b.offset
            return true
        end
    end
    return false
end

# PGF Draw Rectangle
_draw_rect(x0, y0, x1, y1; fill = "black") = """
    \\draw [fill = $fill, $fill, thick]
        (axis cs:$x0,$y0) rectangle (axis cs:$x1,$y1);
    """
function _draw_rect(r::TensorRecord; read_color = :red, write_color = :blue)
    # First, we draw a large black rectangle for the full lifetime of the tensor.
    first = _draw_rect(r.start, r.offset, r.stop, r.offset + r.height)

    # Then, we emit red rectangles for each user. By emitting these afterwards, they should
    # be drawn on top of the the black rectangles.
    reads = map(zip(r.read_starts, r.read_stops)) do x
        start = x[1]
        stop = x[2]
        return _draw_rect(
            start,
            r.offset,
            stop,
            r.offset + r.height;
            fill = read_color
        )
    end
    write = _draw_rect(
        r.start,
        r.offset,
        r.write_stop,
        r.offset + r.height;
        fill = write_color
    )

    return vcat(first, reads, write)
end

function heap_plot(path)
    # Create a record for each tensor in the compiled graph.
    tensor_records = get_records(path)
    sort!(tensor_records)

    # Plot a rectangle for each tensor.
    rectangle_strings = Iterators.flatten(_draw_rect.(tensor_records))

    # Get the axis min and maxes
    ymin = 0
    ymax = maximum(x -> x.offset + x.height, tensor_records)
    xmin = 0
    xmax = maximum(x -> x.start, tensor_records)

    axs = @pgf Axis(
        {
            width = "20cm",
            height = "14cm",
            ymin = ymin,
            ymax = ymax,
            xmin = xmin,
            xmax = xmax,
            xlabel = "Time (s)",
            ylabel = "Memory Position (GB)",
        },
        rectangle_strings...,
    )

    name, _ = splitext(basename(path))

    pgfsave(joinpath(FIGDIR, "$name.pdf"), axs)
    return axs
end

function get_records(path)
    d = deserialize(path)

    tensors = d["tensors"]
    nodes = d["nodes"]

    # Generate aggregate runtimes.
    times = [node["time"] for node in nodes]
    cumsum!(times, times)

    # Convert time from microseconds to seconds
    times = times ./ 1E6
    @show times[end]

    # Add one extra item to the end of `times` so we can index one past the original lenght
    # when calculating stop times of reading nodes.
    #
    # This is not 100% correct but close enough.
    push!(times, times[end])

    # Extract all the things!! :D
    tensor_records = map(tensors) do tensor
        users = tensor["user_indices"]
        user_times = times[users]

        start = first(user_times)
        stop = last(user_times)

        read_starts = user_times
        read_stops = times[users .+ 1]

        write_stop = times[first(users) + 1]

        return TensorRecord(
            start,
            stop,
            tensor["offset"] / 1E9,
            tensor["sizeof"] / 1E9,
            read_starts,
            read_stops,
            write_stop
        )
    end

    return tensor_records
end

#####
##### Plot summary statistics
#####

function barplot(records::OrderedDict{String, <:Dict}, ks;
        modifier = identity,
        reducer = ssum,
        ylabel = "",
        title = "",
    )
    # For each entry in outer dict, take the total sum of the values of interest for each
    # inner dict.
    records = map(collect(keys(records))) do name
        v = records[name]
        return name => Dict(k => reducer(v, k) for k in ks)
    end |> OrderedDict

    # Bar plots to add.
    plots = []

    for (name, data) in records
        # Generate the `x` and `y` coordinates for this sample
        x = replace_.(ks)
        y = [modifier(data[k]) for k in ks]

        # Emit the plot
        push!(plots,
            @pgf(PlotInc(
                Coordinates(x, y),
            )),
            @pgf(LegendEntry(name)),
        )
    end

    axs = @pgf Axis(
        {
            ybar,
            symbolic_x_coords = replace_.(ks),
            enlarge_x_limits = 0.2,
            xtick="data",
            xticklabel_style = {
                rotate = 10,
            },
            ymajorgrids,
            title = titlecase(replace_(title)),
            ylabel = ylabel,
        },
        plots...
    )
    return axs
end
