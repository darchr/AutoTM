# Plot ratios of PMEM to DRAM on the x-axis.
function pgf_speedup(f, ratios::Vector{<:Rational}, cache;
        file = "plot.tex",
        formulations = ("numa", "static", "synchronous")
    )

    paths = canonical_path.(Ref(f), formulations, Ref(cache), Ref(nGraph.Backend("CPU")))
    data = deserialize.(paths)

    pmm_performance = get_pmm_performance(data)
    dram_performance = get_dram_performance(data)

    plots = []
    for (datum, formulation) in zip(data, formulations)
        @show formulation

        # This x and y data point
        x = []
        y = []

        for ratio in ratios
            # Check if we have the "ratio" key
            runs = datum.runs
            if haskey(first(runs), :ratio)
                ind = findfirst(x -> x[:ratio] == ratio, runs)
            else
                @info "Performing Ratio Search Fallback"
                ind = findabsmin(x -> compare_ratio(getratio(x), ratio), datum.runs)
            end

            @show convert(Float64, getratio(datum.runs[ind]))
            @show convert(Float64, ratio)
            @show convert(Float64, compare_ratio(getratio(datum.runs[ind]), ratio))

            perf = pmm_performance / datum.runs[ind][:actual_runtime]
            @show perf

            push!(x, ratio_string(ratio))
            push!(y, perf)
        end

        # Emit the plot for this series.
        append!(plots, [
            @pgf(PlotInc(
                Coordinates(x, y),
            )),
            @pgf(LegendEntry("$formulation")),
        ])
    end

    plt = TikzDocument()
    push!(plt, hasymptote())
    symbolic_coords = ratio_string.(ratios)

    dline = pmm_performance / get_dram_performance(data)

    axs = @pgf Axis(
        {
            ybar,
            enlarge_x_limits=0.30,
            bar_width = "15pt",
            width = "8cm",
            height = "4cm",
            legend_style =
            {
                 at = Coordinate(0.05, 1.05),
                 anchor = "south west",
                 legend_columns = 2
            },
            ymin=0,
            symbolic_x_coords = symbolic_coords,
            nodes_near_coords_align={vertical},
            ymajorgrids,
            ylabel_style={
                align = "center",
            },
            xtick="data",
            ytick = 1:(ceil(Int, pmm_performance / dram_performance)+1),
            ymax = ceil(Int, pmm_performance / dram_performance),
            # Lables
            xlabel = "PMM to DRAM Ratio",
            ylabel = "Speedup over all PMM",

        },
        plots...,
        # Draw a horizontal line at the DRAM performance
        #HLine(pmm_performance / get_dram_performance(data)),
        hline(dline;
              xl = first(symbolic_coords),
              xu = last(symbolic_coords)
             ),
        raw"\addlegendimage{line legend, red, sharp plot, ultra thick}",
        LegendEntry("All DRAM"),
    )

    push!(plt, TikzPicture(axs))

    pgfsave(file, plt)
    return nothing
end

