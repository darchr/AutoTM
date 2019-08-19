#####
##### The plots
#####

_wh() = ("8cm", "6cm")
_ms() = 1.5

function pgf_movement_plot(f, cache, suffix; file = "plot.tex", formulation = "synchronous")
    savefile = canonical_path(f, formulation, cache, nGraph.Backend("CPU"), suffix)
    data = deserialize(savefile)

    # Plot the number of move nodes.
    io_size = data.io_size[]
    dram_sizes = (getname(data.runs, :dram_limit) ./ 1E3) .+ (io_size ./ 1E9)

    x = dram_sizes

    plot = TikzDocument()
    push!(plot, vasymptote())

    plots = [
        @pgf(PlotInc(
             {
                thick,
                mark_options = {
                    scale = _ms(),
                },
             },
             Coordinates(x, getname(data.runs, :bytes_moved_pmem) ./ 1E9)
        )),
        @pgf(LegendEntry("sync DRAM to PMEM")),
        @pgf(PlotInc(
             {
                thick,
                mark_options = {
                    scale = _ms(),
                },
             },
             Coordinates(x, getname(data.runs, :bytes_moved_dram) ./ 1E9)
        )),
        @pgf(LegendEntry("sync PMEM to DRAM")),
    ]

    width, height = _wh()
    axs = @pgf Axis(
        {
            grid = "major",
            xlabel = "DRAM Limit (GB)",
            ylabel = "Memory Moved (GB)",
            width = width,
            height = height,
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
                    mark_options = {
                        scale = _ms(),
                    },
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
                    mark_options = {
                        scale = _ms(),
                    },
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
    # scheme = "Spectral"
    # plotsets = """
    #     \\pgfplotsset{
    #         cycle list/$scheme,
    #         cycle multiindex* list={
    #             mark list*\\nextlist
    #             $scheme\\nextlist
    #         },
    #     }
    # """
    # push!(plot, plotsets)
    push!(plot, vasymptote())
    width, height = _wh()

    axs = @pgf Axis(
        {
            grid = "major",
            width = width,
            height = height,
            xlabel = "DRAM Limit (GB)",
            ylabel = "Percent of Kernel IO in DRAM",
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

function pgf_plot_performance(f, cache, suffix;
        file = "plot.tex",
        formulations = ("static", "synchronous",)
    )

    paths = canonical_path.(
        Ref(f), 
        formulations, 
        Ref(cache), 
        Ref(nGraph.Backend("CPU")),
        Ref(suffix),
    )
    data = deserialize.(paths)

    pmm_performance = get_pmm_performance(data)
    dram_performance = get_dram_performance(data)

    plots = []
    for (datum, formulation) in zip(data, formulations)
        io_size = datum.io_size[]
        dram_sizes = (getname(datum.runs, :dram_limit) ./ 1E3) .+ (io_size ./ 1E9)

        x = dram_sizes
        y = getname(datum.runs, :actual_runtime) ./ dram_performance

        append!(plots, [
            @pgf(PlotInc(
                {
                   thick,
                   mark_options = {
                       scale = _ms(),
                   },
                },
                Coordinates(dram_sizes, y))
            ),
            @pgf(LegendEntry(formulation))
        ])
    end

    plot = TikzDocument()
    width, height = _wh()
    axs = @pgf Axis(
        {
            width = width,
            height = height,
            grid = "major",
            xlabel = "DRAM Limit (GB)",
            ylabel = "Performance Relative\\\\to all DRAM",
            # put legend on the bottom right
            legend_style = {
                at = Coordinate(1.0, 1.0),
                anchor = "north east",
            },
        },
        plots...
    )

    push!(plot, TikzPicture(axs))

    empty!(PGFPlotsX.CUSTOM_PREAMBLE)
    push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usepgfplotslibrary{colorbrewer}")

    pgfsave(file, plot)
end
