
# replace underscores with spaces to LaTeX is happy
replace_(x) = replace(String(x), "_" => " ")
sumby(v, n) = [sum(@view(v[i:min(i+n-1, length(v))])) for i in 1:n:length(v)]
ssum(x, s) = sum(sum.(x[s]))

# Main workhorse function.
function make_plot(
        data::SocketCounterRecord,
        names;
        sumover = 1,
        plotsum = false,
        xlabel = "Time (s)",
        ylabel = "",
        ymax = nothing,
        modifier = identity,
        title = "",
        reducer = sum,
    )

    selected_data = Dict(k => reducer.(retrieve(data, k)) for k in names)

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

# Look in the `data` directory - deserialize all files that match `prefix`.
# Furthermore - do a pairwise merging of the named tuple entries as long as the difference
# in lengths between the everything is with `max_truncate`.
function load(
        file,
        params = NamedTuple();
        # keyword arguments
        f = x -> true,
        max_truncate = 5,
        socket = 2,
        show = false,
    )

    database = deserialize(joinpath(DATADIR, file))

    # Filter out entries that match the request.
    filtered_database = database[params]

    if length(filtered_database) == 0
        println(database)
        throw(error("No results matching your query!"))
    elseif length(filtered_database) > 1
        show_different(filtered_database)
        throw(error("Found $(length(filtered_database)) entries"))
    end

    # Get the actual payload from the data.
    data = first(filtered_database[:data])[Symbol("socket_$(socket-1)")]::SocketCounterRecord

    # Check if number of samples is about the same.
    min, max = extrema(length, walkleaves(data))
    if max - min > max_truncate
        error("Sizes of data not within $max_truncate. Sizes are: $(length.(data))")
    end

    # Funtion to resize all arrays
    myresize! = (x, y) -> resize!(last(x), y)
    myresize!.(walkleaves(data), min)
    return data
end

# Helper function to highlight the difference between entries.
function show_different(database::DataTable{names}) where {names}
    # Find all names where all values are not equal
    mismatch_names = Symbol[]
    for name in names
        x = database[name]
        if !all(isequal(first(x)), x)
            push!(mismatch_names, name)
        end
    end

    # Now that we have the mismatched names, we can create a new datatable with just those
    # entries.
    params = Any[]
    for row in Tables.rows(database)
        param = NamedTuple{Tuple(mismatch_names)}(Tuple(getproperty.(Ref(row), mismatch_names)))
        push!(params, param)
    end

    d = DataTable()
    for p in params
        addentry!(d, p)
    end
    println(d)
end

