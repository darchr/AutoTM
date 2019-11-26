
# replace underscores with spaces to LaTeX is happy
replace_(x) = replace(String(x), "_" => " ")
sumby(v, n) = [sum(@view(v[i:min(i+n-1, length(v))])) for i in 1:n:length(v)]
ssum(x, s) = sum(sum.(x[s]))

# Main workhorse function for line plots
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
        linewidth = "4pt",
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
                    line_width = linewidth,
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

labelize(x) = x
labelize(x::String) = replace_(x)
labelize(x::Tuple) = join(labelize.(x), " ")

# Now bar plots!
function bar_plot(
        # The actual data for each item of interest.
        data::Vector{<:SocketCounterRecord},
        # The names of the counters to retrieve,
        counters,
        # The series lables
        labels;
        xlabel = "",
        ylabel = "",
        title = "",
        aggregator = sum,
        reducer = sum,
        ymax = nothing,
        width = "8cm",
        height = "8cm",
    )

    @assert length(data) == length(labels)

    # First step - pull out the data we want into dictionary form.
    selected_data = map(data) do d
        # For each counters, aggregate the counters according to `aggregate`,
        # then reduce the resulting vector to a single number using `reducer`.
        return map(counters) do counter
            aggregated = aggregator.(retrieve(d, counter))
            @show sum(aggregated)
            reduced_data = reducer(aggregated)
            return counter => reduced_data
        end |> Dict
    end

    # Plot the results
    plots = []
    for counter in counters
        y_coords = getindex.(selected_data, counter)
        x_coords = labelize.(labels)

        coords = Coordinates(x_coords, y_coords)
        push!(plots,
            @pgf(PlotInc(coords)),
            @pgf(LegendEntry(replace_(counter)))
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
            width = width,
            height = height,
            ultra_thick,
            ybar,
            ymin = 0,
            legend_style = {
                at = Coordinate(0.98, 1.02),
                anchor = "south east",
            },
            legend_columns = 2,
            grid = "both",

            # Set up symbolic coordinates.
            symbolic_x_coords = labelize.(labels),
            nodes_near_coords_align={vertical},
            xtick = "data",
            xticklabel_style = {
                rotate = -15,
            },
            #enlarge_x_limits = 0.3,

            # Setup labels
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
        max_truncate = 5,
        socket = 2,
    )

    database = deserialize(joinpath(DATADIR, file))

    # Filter out entries that match the request.
    filtered_database = database[params]
    if length(filtered_database) == 0
        println(database)
        throw(error("No results matching your query!"))
    elseif length(filtered_database) > 1
        println(filtered_database)
        throw(error("Found $(length(filtered_database)) entries"))
    end

    # Get the actual payload from the data.
    data = first(filtered_database).data[Symbol("socket_$(socket-1)")]::SocketCounterRecord

    # Check if number of samples is about the same.
    min, max = extrema(length, walkleaves(data))
    if max - min > max_truncate
        error("Sizes of data not within $max_truncate. Sizes are between $min and $max.")
    end

    # Funtion to resize all arrays
    resize!.(walkleaves(data), min)
    return data
end

function load_multiple(
        file,
        params::Union{<:NamedTuple, <:Vector},
        # The field that should be different across the selected matches
        should_vary;
        socket = 2,
    )

    db = deserialize(joinpath(DATADIR, file))

    # Filter out entries that match the request.
    #
    # If `params` is a NamedTuple, find all matches.
    # If `params` is a vector of named tuples, find a match for each.
    if isa(params, NamedTuple)
        filtered_database = db[params]
    elseif isa(params, Vector)
        filtered_database = Slab()
        for p in params
            @show p
            x = db[p]
            if length(x) > 1
                println(x)
                error()
            end
            filtered_database[p] = x[1].data
        end
    end

    if isnothing(filtered_database)
        println(db)
        throw(error("No results matching your query!"))
    end

    # Ensure that the `should_vary` field of the filtered_database does infact have
    # all unique entries.
    @assert allunique(filtered_database[should_vary])

    # Now, extract the socket records and return!
    key = Symbol("socket_$(socket - 1)")
    data = [entry.data[key]::SocketCounterRecord for entry in filtered_database]
    return data, filtered_database[should_vary]
end
