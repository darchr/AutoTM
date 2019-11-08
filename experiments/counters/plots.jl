module CounterPlots

# replace underscores with spaces to LaTeX is happy
replace_(x) = replace(String(x), "_" => " ")
sumby(v, n) = [sum(@view(v[i:min(i+n-1, length(v))])) for i in 1:n:length(v)]

include("Traffic.jl")
using Serialization
using Statistics
using Dates
using DataStructures
using StructArrays

using PGFPlotsX

const DATADIR = joinpath(@__DIR__, "data")
const FIGDIR = joinpath(@__DIR__, "figures")

# Make the figure directory if needed.
function __init__()
    isdir(FIGDIR) || mkdir(FIGDIR)
end

ssum(x, s) = sum(sum.(x[s]))

function make_plot(
        data, names;
        sumover = 1,
        plotsum = false,
        xlabel = "Time (s)",
        ylabel = "",
        ymax = nothing,
        modifier = identity,
        title = "",
    )

    selected_data = Dict(k => sum.(data[k]) for k in names)
    if plotsum
        selected_data[:total_sum] = map(sum, zip(values(selected_data)...))
        push!(names, :total_sum)
    end
    _sums = Dict(k => sum(v) for (k,v) in selected_data)
    display(_sums)

    # Plot the results
    plots = []
    for name in names
        v = selected_data[name]
        coords = Coordinates(1:sumover:length(v), sumby(modifier(v), sumover))
        push!(plots,
            @pgf(PlotInc(coords)),
            @pgf(LegendEntry(replace_(name)))
        )
    end

    empty!(PGFPlotsX.CUSTOM_PREAMBLE)
    push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usepgfplotslibrary{colorbrewer}")
    plt = TikzDocument()
    push!(plt, "\\pgfplotsset{cycle list/Dark2}")

    tikz = TikzPicture()

    opts = []
    isnothing(ymax) || push!(opts, "ymax = $ymax")

    axs = @pgf Axis(
        {
            width = "12cm",
            height = "10cm",
            ultra_thick,
            legend_style = {
                at = Coordinate(0.02, 0.98),
                anchor = "north west",
            },
            grid = "major",
            xlabel = xlabel,
            ylabel = ylabel,
            title = title,
            opts...
        },
        plots...
    )
    push!(tikz, axs)
    push!(plt, tikz)

    return plt
end

# The structure of this data is the WORST thing ever.
#
# Every serialized file has the following structure
#
# NamedTuple
# :timestamp -> Vector{DateTime}
# :counters -> Vector{Tuple}
#     index 1 -> socket 0
#         NamedTuple of Measurements
#     index 2 -> socket 1
#         NamedTuple of Measurements

function counters_for_socket(x, i)
    array_of_nt = getindex.(x.counters, i)
    # Convert to dictionary - make life easier on ourselves
    return Dict(n => getproperty.(array_of_nt, n) for n in keys(first(array_of_nt)))
end

isscratchpad(x) = occursin("scratchpad", x)

# Look in the `data` directory - deserialize all files that match `prefix`.
# Furthermore - do a pairwise merging of the named tuple entries as long as the difference
# in lengths between the everything is with `max_truncate`.
function load(
        prefix,
        nt = NamedTuple();
        # keyword arguments
        f = x -> true,
        max_truncate = 5,
        socket = 2,
        dir = DATADIR,
        show = false,
    )

    # Get all files matching the prefix and possibly the suffix and deserialize
    files = filter(x -> startswith(x, prefix) && f(x), readdir(dir))
    for (k,v) in pairs(nt)
        str = "_$(Traffic.modify(k, v))_"
        filter!(x -> occursin(str, x), files)
    end

    show && (@show files)

    data = StructArray.(deserialize.(joinpath.(dir, files)))

    # Check if number of samples is about the same.
    min, max = extrema(length, data)
    if max - min > max_truncate
        error("Sizes of data not within $max_truncate. Sizes are: $(length.(data))")
    end
    resize!.(data, min)
    counters = counters_for_socket.(data, socket)
    return reduce(merge, counters)
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

end
