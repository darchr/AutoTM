function pgf_error_plot(fns, ratios, caches; 
        file = "plot.tex", 
        formulations = ("static", "synchronous"),
        suffix = nothing
    )
    
    plots = []
    backend = nGraph.Backend("CPU")
    for formulation in formulations
        for f in fns
            for cache in caches
                x = []
                y = []
                data = deserialize(canonical_path(f, formulation, cache, backend, suffix))

                for ratio in ratios
                    ind = findabsmin(x -> compare_ratio(getratio(x), ratio), data.runs)
                    actual_runtime = data.runs[ind][:actual_runtime]
                    predicted_runtime = data.runs[ind][:predicted_runtime] / 1E6
                    push!(x, ratio_string(ratio))
                    push!(y, 100 * (predicted_runtime - actual_runtime) / actual_runtime)
                end

                append!(plots, [
                    @pgf(PlotInc(Coordinates(x, y)))
                    @pgf(LegendEntry("$(titlename(f)) - $formulation"))
                ])
            end
        end
    end

    # Make the plot itself
    plt = TikzDocument()
    push!(plt, """
    \\pgfplotsset{
        width=8cm,
        height=4cm}
    """)

    axs = @pgf Axis(
        {
            ybar,
            bar_width = "2pt",
            grid = "major",
            xlabel = "DRAM Limit (GB)",
            ylabel = "Relative Predicted\\\\Runtime Error \\%",
            ylabel_style={
                align = "center",
            },
            # Put the legend outside on the right
            legend_style = {
                at = Coordinate(0.50, 1.20),
                anchor = "south",
                legend_columns = 2,
            },
            legend_cell_align = {"left"},

            # Setup x coordinates
            symbolic_x_coords = ratio_string.(ratios), 
            xtick = "data",
        },
        plots...
    )

    push!(plt, TikzPicture(axs))
    pgfsave(file, plt)

    return nothing
end
