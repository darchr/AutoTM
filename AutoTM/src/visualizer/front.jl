# THe front plot
function plot_front(fns, ratio::Rational; 
        formulations = ("numa", "synchronous"),
        file = "plot.tex",
    )

    bar_plots = []

    pmm_runtimes = Dict{Any,Float64}()
    dram_runtimes = Dict{Any,Float64}()
    for f in fns
        baselines = load_save_files(f, formulations) 
        pmm_runtimes[f] = get_pmm_performance(baselines)
        dram_runtimes[f] = get_dram_performance(baselines)
    end

    for formulation in formulations
        x = []
        y = []

        for (i, f) in enumerate(fns)
            datum = load_save_files(f, formulation)
            ind = findabsmin(x -> compare_ratio(getratio(x), ratio), datum.runs)
            perf = pmm_runtimes[f] / datum.runs[ind][:actual_runtime]

            push!(x, i)
            push!(y, perf)
        end

        append!(bar_plots, [
            @pgf(PlotInc(Coordinates(x, y)))
            @pgf(LegendEntry(formulation))
        ])
    end

    # Add DRAM performance
    x = []
    y = []
    for (i, f) in enumerate(fns)  
        push!(x, i)
        push!(y, pmm_runtimes[f] / dram_runtimes[f])
    end
    append!(bar_plots, [
        @pgf(PlotInc(Coordinates(x, y)))
        @pgf(LegendEntry("dram"))
    ])

    plt = TikzDocument()
    push!(plt, """
    \\pgfplotsset{width=7cm,height=4cm}
    """)

    xticklabels = collect(titlename.(fns))
    @show xticklabels

    # Bar axis
    tikz = TikzPicture() 
    axs = @pgf Axis(
        {
            ybar,
            bar_width = "10pt",
            enlarge_x_limits=0.30,

            legend_style =
            {
                 at = Coordinate(0.05, 1.05),
                 anchor = "south west",
                 legend_columns = -1
            },
            ymin=0,
            ymajorgrids,
            ylabel_style={
                align = "center",
            },

            # Lables
            ylabel = "Speedup over all PMEM",

            xtick = "data",
            xticklabels = xticklabels,
            xticklabel_style={
                align = "center",
            },
        },
        bar_plots...,
    )
    push!(tikz, axs)
    push!(plt, tikz)

    pgfsave(file, plt)
    return nothing
end
