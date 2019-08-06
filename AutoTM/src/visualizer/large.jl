function pgf_large_performance(fns; 
        file = "plot.tex", 
        formulations = ("static", "synchronous")
    )
    # Step through each function, get the 2lm performance
    bar_plots = []
    coords = String[]

    baseline_runtime = Dict{Any,Float64}()
    for f in fns
        baseline = load_save_files(f, "2lm")
        baseline_runtime[f] = minimum(getname(baseline.runs, :actual_runtime))

        push!(coords, "$(titlename(f)) ($(f.batchsize))")
    end

    for formulation in formulations
        x = []
        y = []

        for (i, f) in enumerate(fns)
            datum = load_save_files(f, formulation)
            speedup = baseline_runtime[f] / minimum(getname(datum.runs, :actual_runtime))
            push!(x, "$(titlename(f)) ($(f.batchsize))")
            push!(y, speedup)
        end

        append!(bar_plots, [
            @pgf(PlotInc(
                Coordinates(x, y),
            ))
            @pgf(LegendEntry(formulation))
        ])
    end

    plt = TikzDocument()
    push!(plt, """
    \\pgfplotsset{width=7cm,height=4cm}
    """)

    # Bar axis
    tikz = TikzPicture() 
    axs = @pgf Axis(
        {
            ybar,
            bar_width = "20pt",
            enlarge_x_limits=0.30,
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
                rotate = 15,
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
