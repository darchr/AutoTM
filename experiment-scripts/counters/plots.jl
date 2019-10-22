module CounterPlots

# replace underscores with spaces to LaTeX is happy
replace_(x) = replace(String(x), "_" => " ")
sumby(v, n) = [sum(@view(v[i:min(i+n-1, length(v))])) for i in 1:n:length(v)]

using Serialization
using Statistics
using Dates

using PGFPlotsX
using AutoTM

const DATADIR = joinpath(@__DIR__, "data")

function make_plot(
        data, names;
        sumover = 1,
        plotsum = false,
        normalize = false,
        xlabel = "Time (s)",
        ylabel = "",
        ymax = 90,
    )
    # Add 1 to the socket to adjust for
    socket  += 1

    # Get the values for each counter over time.
    accumulators = Dict{Symbol,Vector{Float64}}()
    for sample in counters
        # Iterate over measurement types
        sum_accumulator = 0.0
        for counter_name in names
            channel_values = sample[counter_name]

            # Initialize if needed
            A = get!(accumulators, counter_name, Float64[])
            push!(A, sum(channel_values))

            # Also keep a rolling sum of ALL counters for plotting if desired
            sum_accumulator += last(A)
        end
        if plotsum
            A = get!(accumulators, :total_sum, Float64[])
            push!(A, sum_accumulator)
        end

        # If normalizing, accumulate across the items we just added and normalize them.
        # Preload the keys so we don't accidentally change the iteration order of the dict.'
        if normalize
            total = 0.0
            for v in values(accumulators)
                total += v[end]
            end
            for v in values(accumulators)
                v[end] = v[end] / total
            end
        end
    end

    # Print out some statistics
    for (k,v) in accumulators
        println("$k total: ", sum(v))
        println("$k mean: ", mean(v))
    end


    # Plot the results
    plots = []
    ks = sort(collect(keys(accumulators)))
    for k in ks
        v = accumulators[k]

        # If we're normalizing the output, don't try to convert DRAM/PMM read/write commands
        # into a bandwidth.
        if normalize
            coords = Coordinates(1:sumover:length(v), sumby(v, sumover))
        else
            coords = Coordinates(1:sumover:length(v), sumby(v.* 64 ./ (1E9 .* sumover), sumover))
        end

        push!(plots,
            @pgf(PlotInc(coords)),
            @pgf(LegendEntry(replace_(k)))
        )
    end

    plt = TikzDocument()
    empty!(PGFPlotsX.CUSTOM_PREAMBLE)
    push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usepgfplotslibrary{colorbrewer}")

    push!(plt, "\\pgfplotsset{cycle list/Dark2}")

    tikz = TikzPicture()

    axs = @pgf Axis(
        {
            #colorbrewer_cycle_list="Set1",
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
            ymax = ymax,
        },
        plots...
    )
    push!(tikz, axs)
    push!(plt, tikz)

    pgfsave("temp.tex", plt)
    return plt
end

# The structure of this data is the worst thing ever.
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

# Methods for querying and dealing with SystemSnoop style data.
snoop_length(x::NamedTuple) = minimum(length, x)
snoop_truncate(x, n) = resize!.(x, n)
snoop_truncate(x::NamedTuple, n) = snoop_truncate(Tuple(x), n)

function counters_for_socket(x::NamedTuple, i)
    array_of_nt = getindex.(x.counters, i)
    # Convert to dictionary - make life easier on ourselves
    return Dict(n => getproperty.(array_of_nt, n) for n in keys(first(array_of_nt)))
end

# Look in the `data` directory - deserialize all files that match `prefix`.
# Furthermore - do a pairwise merging of the named tuple entries as long as the difference
# in lengths between the everything is with `max_truncate`.
function load(prefix; max_truncate = 5, socket = 2)
    # Get all files matching the prefix and deserialize
    files = filter(x -> startswith(x, prefix), readdir(DATADIR))
    data = deserialize.(joinpath.(DATADIR, files))

    # Check if number of samples is about the same.
    min, max = extrema(snoop_length, data)
    if max - min > max_truncate
        error("Sizes of data not within $max_truncate")
    end
    snoop_truncate.(data, min)
    counters = counters_for_socket.(data, socket)
    return reduce(merge, counters)

    return reduce((x...) -> merge.(counters_for_socket.(x, socket)...), data)
end

end
