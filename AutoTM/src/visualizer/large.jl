function pgf_large_performance(fns, cache, cache_2lm; 
        file = "plot.tex", 
        formulations = ("static", "synchronous")
    )
    # Step through each function, get the 2lm performance
    bar_plots = []
    coords = String[]
    backend = nGraph.Backend("CPU")

    baseline_runtime = Dict{Any,Float64}()
    for f in fns
        baseline = deserialize(canonical_path(f, "2lm", cache_2lm, backend))
        baseline_runtime[f] = minimum(getname(baseline.runs, :actual_runtime))

        push!(coords, "$(titlename(f)) ($(f.batchsize))")
    end

    for formulation in formulations
        x = []
        y = []

        for (i, f) in enumerate(fns)
            data = deserialize(canonical_path(f, formulation, cache, backend))
            speedup = baseline_runtime[f] / minimum(getname(data.runs, :actual_runtime))
            push!(x, "$(titlename(f)) ($(f.batchsize))")
            push!(y, speedup)
        end

        append!(bar_plots, [
            @pgf(PlotInc(
                Coordinates(x, y),
            ))
            @pgf(LegendEntry(get(_speedup_formulation_lut, formulation, formulation)))
        ])
    end

    plt = TikzDocument()
    push!(plt, """
    \\pgfplotsset{width=10cm,height=4cm}
    """)

    # Bar axis
    tikz = TikzPicture() 
    axs = @pgf Axis(
        {
            ybar,
            bar_width = "18pt",
            enlarge_x_limits=0.20,
            symbolic_x_coords = coords,
            #nodes_near_coords_align={vertical},

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
            xticklabel_style={
                rotate = 10,
            },
            xtick = "data",

            # Lables
            ylabel = "Speedup over 2LM",
        },
        bar_plots...,
    )
    push!(tikz, axs)
    push!(plt, tikz)

    pgfsave(file, plt)
    return nothing
end
