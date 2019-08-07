#####
##### The plots
#####

function pgf_movement_plot(f, cache, suffix; file = "plot.tex", formulation = "synchronous")
    savefile = canonical_path(f, formulation, cache, nGraph.Backend("CPU"), suffix)
    data = deserialize(savefile)

    # Plot the number of move nodes.
    io_size = data.io_size[]
    dram_sizes = (getname(data.runs, :dram_limit) ./ 1E3) .+ (io_size ./ 1E9)

    x = dram_sizes

    plot = TikzDocument()
    scheme = "Spectral"
    plotsets = """
        \\pgfplotsset{
            cycle list/$scheme,
            cycle multiindex* list={
                mark list*\\nextlist
                $scheme\\nextlist
            },
        }
    """
    push!(plot, plotsets)
    push!(plot, vasymptote())

    plots = [
        @pgf(PlotInc(
             {
                thick,
             },
             Coordinates(x, getname(data.runs, :bytes_moved_pmem) ./ 1E9)
        )),
        @pgf(LegendEntry("sync DRAM to PMEM")),
        @pgf(PlotInc(
             {
                thick,
             },
             Coordinates(x, getname(data.runs, :bytes_moved_dram) ./ 1E9)
        )),
        @pgf(LegendEntry("sync PMEM to DRAM")),
    ]


    axs = @pgf Axis(
        {
            grid = "major",
            xlabel = "DRAM Limit (GB)",
            ylabel = "Memory Moved (GB)",
            vasymptote = data.default_alloc_size[] / 1E9,
        },
        plots...,
    )

    push!(plot, TikzPicture(axs))

    empty!(PGFPlotsX.CUSTOM_PREAMBLE)
    push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usepgfplotslibrary{colorbrewer}")

    pgfsave(file, plot)

    return nothing
end

function pgf_io_plot(f, cache, suffix; file = "plot.tex", formulations = ("synchronous",))
    savefiles = canonical_path.(
        Ref(f),
        formulations,
        Ref(cache),
        Ref(nGraph.Backend("CPU")),
        Ref(suffix)
    )
    data = deserialize.(savefiles)

    # Plot the number of move nodes.
    io_size = first(data).io_size[]
    plots = []
    for (d, formulation) in zip(data, formulations)
        dram_sizes = (getname(d.runs, :dram_limit) ./ 1E3) .+ (io_size ./ 1E9)
        push!(plots, @pgf(
            PlotInc(
                 {
                    thick,
                 },
                 Coordinates(
                    dram_sizes,
                    getname(d.runs, :bytes_dram_input_tensors) ./ getname(d.runs, :bytes_input_tensors)
                )
            ),
        ))
        push!(plots, @pgf(LegendEntry("$formulation: input tensors")))
        push!(plots, @pgf(PlotInc(
                 {
                    thick,
                 },
                 Coordinates(
                    dram_sizes,
                    getname(d.runs, :bytes_dram_output_tensors) ./ getname(d.runs, :bytes_output_tensors)
                 )
            ),
        ))
        push!(plots, @pgf(LegendEntry("$formulation: output tensors")))
    end

    plot = TikzDocument()
    scheme = "Spectral"
    plotsets = """
        \\pgfplotsset{
            cycle list/$scheme,
            cycle multiindex* list={
                mark list*\\nextlist
                $scheme\\nextlist
            },
        }
    """
    push!(plot, plotsets)
    push!(plot, vasymptote())

    axs = @pgf Axis(
        {
            grid = "major",
            xlabel = "DRAM Limit (GB)",
            ylabel = "Percent of Kernel Arguments in DRAM",
            # put legend on the bottom right
            legend_style = {
                at = Coordinate(1.0, 0.0),
                anchor = "south east",
            },
            vasymptote = first(data).default_alloc_size[] ./ 1E9,
        },
        plots...
    )

    push!(plot, TikzPicture(axs))

    empty!(PGFPlotsX.CUSTOM_PREAMBLE)
    push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usepgfplotslibrary{colorbrewer}")

    pgfsave(file, plot)

    return nothing
end

