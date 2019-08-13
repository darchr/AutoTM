function pgf_price_performance(pairs::Vector{<:Pair}, ratios::Vector{<:Rational}; 
        cost_ratio = 2.1,
        file = "plot.tex", 
    )

    plots = [] 

    # Cost Axis
    y_cost = map(ratios) do ratio
        # PMM
        num = ratio.num

        # DRAM
        den = ratio.den

        # Total approximately sums to 1
        return (num / cost_ratio + den) / (num + den)
    end

    for (f, formulation) in pairs
        data = load_save_files(f, formulation)
        dram_performance = get_dram_performance(data)

        # This x and y data point
        x = []
        y = []

        for (i, ratio) in enumerate(ratios)
            ind = findabsmin(x -> compare_ratio(getratio(x), ratio), data.runs)
            perf = dram_performance / data.runs[ind][:actual_runtime]

            push!(x, ratio_string(ratio))
            push!(y, y_cost[i] / perf)
        end

        # Emit the plot for this series.
        append!(plots, [
            @pgf(PlotInc(
                Coordinates(x, y),     
            )),
            @pgf(LegendEntry(replace(titlename(f), "_" => " "))),
        ])
    end

    plt = TikzDocument()
    push!(plt, """
    \\pgfplotsset{width=7cm,height=4cm}
    """)

    symbolic_coords = ratio_string.(ratios)

    # Bar axis
    tikz = TikzPicture() 
    axs = @pgf Axis(
        {
            ybar,
            bar_width = "5pt",
            legend_style =
            {
                 at = Coordinate(-0.05, 1.05),
                 anchor = "south west",
                 legend_columns = 2,
            },
            ymin = 0,
            symbolic_x_coords = symbolic_coords,
            nodes_near_coords_align={vertical},
            ymajorgrids,
            ylabel_style={
                align = "center",
            },
            xtick="data",
            #ytick = [0,1],
            # Lables
            xlabel = "PMM to DRAM Ratio",
            ylabel = "Normalize\\\\Price-Performance Ratio",

        },
        plots...,
    )
    push!(tikz, axs)
    push!(plt, tikz)

    pgfsave(file, plt)
    return nothing
end
