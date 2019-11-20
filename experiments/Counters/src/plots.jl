
# replace underscores with spaces to LaTeX is happy
replace_(x) = replace(String(x), "_" => " ")
sumby(v, n) = [sum(@view(v[i:min(i+n-1, length(v))])) for i in 1:n:length(v)]
ssum(x, s) = sum(sum.(x[s]))

# Main workhorse function.
function make_plot(
        data, names;
        sumover = 1,
        plotsum = false,
        xlabel = "Time (s)",
        ylabel = "",
        ymax = nothing,
        modifier = identity,
        title = "",
        reducer = sum,
    )

    selected_data = Dict(k => reducer.(data[k]) for k in names)
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
            @pgf(PlotInc(
                {
                    line_width="4pt",
                },
                coords
            )),
            @pgf(LegendEntry(replace_(name)))
        )
    end

    empty!(PGFPlotsX.CUSTOM_PREAMBLE)
    push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usepgfplotslibrary{colorbrewer}")
    plt = TikzDocument()
    push!(plt, "\\pgfplotsset{cycle list/Paired}")

    tikz = TikzPicture()

    opts = []
    isnothing(ymax) || push!(opts, "ymax = $ymax")

    axs = @pgf Axis(
        {
            width = "8cm",
            height = "8cm",
            ultra_thick,
            legend_style = {
                at = Coordinate(0.02, 1.02),
                anchor = "south west",
            },
            legend_columns = 2,
            grid = "both",
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
#
# To make matters worse, we now have to deal with both Core and Uncore counters.
#
# Add a layer of indirection for handling the difference.
peel(x::Tuple, i) = x[i]
peel(x::NamedTuple, i) = x

function counters_for_socket(x, i)
    array_of_nt = peel.(x.counters, i)
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
        str = "_$(modify(k, v))"
        filter!(x -> occursin(str, x), files)
    end

    if length(files) == 0
        throw(error("No files found matching query!"))
    elseif length(files) > 1
        str = "Found $(length(files)) files!"
        errmsg = join((str, files...), "\n")
        throw(error(errmsg))
    end
    @assert length(files) == 1
    data = StructArray.(Iterators.flatten(deserialize.(joinpath.(dir, files))))

    # Check if number of samples is about the same.
    min, max = extrema(length, data)
    if max - min > max_truncate
        error("Sizes of data not within $max_truncate. Sizes are: $(length.(data))")
    end
    resize!.(data, min)
    counters = counters_for_socket.(data, socket)
    return reduce(merge, counters)
end

