#####
##### Plots for GPU code
#####

function pgf_gpu_performance_plot(
        funcs, cache;
        file = "plot.tex",
        formulations = ("synchronous", "asynchronous")
    )

    backend = nGraph.Backend("GPU")

    bar_plots = []
    memory_plots = []
    annotations = []

    coords = []
    batchsize_offset = -0.5
    network_offset = -1.5

    # First step, compute the managed runtime performance for each function as a baseline
    baseline_runtime = Dict{Any,Float64}()
    for (i, f) in enumerate(funcs)
        for formulation in formulations
            data = deserialize(canonical_path(f, formulation, cache, backend))
            baseline_runtime[f] = min(
                get(baseline_runtime, f, typemax(Float64)),
                data.gpu_managed_runtime[]
            )
        end
        push!(annotations, comment(i, batchsize_offset, f.batchsize))
    end

    # Generate function labels
    for title in unique(titlename.(funcs))
        # Get the start and stop indices for this title
        start = findfirst(x -> titlename(x) == title, funcs)
        stop = findlast(x -> titlename(x) == title, funcs)
        center = (start + stop) / 2

        push!(annotations, comment(center, network_offset, title))
    end

    # Generate data for the formulations
    for formulation in formulations
        y = []
        x = []

        for (i, f) in enumerate(funcs)
            data = deserialize(canonical_path(f, formulation, cache, backend))
            speedup = baseline_runtime[f] / minimum(getname(data.runs, :actual_runtime))

            push!(x, i)
            push!(y, speedup)

        end
        append!(bar_plots, [
            @pgf(PlotInc(
                Coordinates(x, y),
            ))
            @pgf(LegendEntry(formulation))
        ])
    end

    # Finally, plot the ideal implmenetation
    x = []
    y = []
    for (i, f) in enumerate(funcs)
        data = deserialize.(canonical_path.(Ref(f), formulations, Ref(cache), Ref(backend)))

        # Conovert milliseconds to seconds
        ideal = minimum(minimum.(getname.(getname(data, :runs), :oracle_time))) / 1E6
        ideal = min(ideal, baseline_runtime[f])
        @show ideal

        push!(x, i)
        push!(y, baseline_runtime[f] / ideal)
    end

    append!(bar_plots, [
        @pgf(PlotInc(
            Coordinates(x, y),
        ))
        @pgf(LegendEntry("oracle"))
    ])

    # Generate the lefthand bar plot
    plt = TikzDocument()
    push!(plt, """
    \\pgfplotsset{
        width=12cm,
        height=5cm
    }
    """)

    tikz = TikzPicture()
    axs = @pgf Axis(
        {
            ybar,
            xmin = 0.5,
            xmax = length(funcs) + 0.5,
            xmajorticks = "false",
            bar_width = "5pt",
            clip = "false",
            legend_style =
            {
                 at = Coordinate(0.05, 1.05),
                 anchor = "south west",
                 legend_columns = -1
            },
            ymin=0,
            ymajorgrids,
            ytick = 1:8,
            ylabel_style={
                align = "center",
            },

            # Lables
            ylabel = "Speedup over\\\\CudaMallocManaged",
        },
        bar_plots...,
        annotations...,
    )
    push!(tikz, axs)
    push!(plt, tikz)

    pgfsave(file, plt)
    return nothing
end
