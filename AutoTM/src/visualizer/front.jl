iterations_per_hour(runtime) = 3600 / (runtime)

#####
##### The actual front plot.
#####

const LEGEND_LUT = Dict(
    "synchronous" => "\\textbf{AutoTM}",
    "numa" => "NUMA",
)
_legend(x) = get(LEGEND_LUT, x, x)

function plot_front(f, ratio::Rational, cache;
        formulations = ("numa", "synchronous"),
        file = "plot.tex",
    )

    bar_plots = []

    backend = nGraph.Backend("CPU")
    data = deserialize.(canonical_path.(Ref(f), formulations, Ref(cache), Ref(backend)))

    pmm_iterations = iterations_per_hour(get_pmm_performance(data))
    dram_iterations = iterations_per_hour(get_dram_performance(data))

    @show pmm_iterations
    @show dram_iterations

    tick_offset = -10
    horizontal = 1
    barwidth = "20pt"

    # Add the PMM performance
    append!(bar_plots, [
        @pgf(PlotInc(
            {
                bar_shift = 0,
            },
            Coordinates([horizontal], [pmm_iterations]))
        ),
        comment(horizontal, tick_offset, "All PMM"),
    ])

    # Add performance for the inner formuliations
    horizontal += 1
    for formulation in formulations
        x = []
        y = []

        data = deserialize(canonical_path(f, formulation, cache, backend))
        ind = findabsmin(x -> compare_ratio(getratio(x), ratio), data.runs)
        perf = iterations_per_hour(data.runs[ind][:actual_runtime])

        @show formulation
        @show perf

        append!(bar_plots, [
            @pgf(PlotInc(
                {
                    bar_shift = 0,
                },
                Coordinates([horizontal], [perf]))
            ),
            comment(horizontal, tick_offset, _legend(formulation)),
        ])
        horizontal += 1
    end

    # Add DRAM performance
    append!(bar_plots, [
        @pgf(PlotInc(
            {
               bar_shift = 0,
            },
            Coordinates([horizontal], [dram_iterations]))
        ),
        comment(horizontal, tick_offset, "All DRAM")
    ])

    plt = TikzDocument()
    push!(plt, """
    \\pgfplotsset{width=12cm,height=5cm}
    """)

    #xticklabels = collect(titlename.(fns))
    #@show xticklabels

    # Bar axis
    comment_level = 110
    tikz = TikzPicture()
    font = "\\scriptsize"
    ymax = 130

    axs = @pgf Axis(
        {
            ybar,
            bar_width = barwidth,
            clip = "false",
            xmajorticks = "false",
            enlarge_x_limits=0.20,

            legend_style =
            {
                 at = Coordinate(0.15, 1.05),
                 anchor = "south west",
                 legend_columns = -1
            },
            ymin=0,
            ymax = ymax,
            ymajorgrids,
            ytick = [0, 40, 80],
            ylabel_style={
                align = "center",
            },

            # Lables
            ylabel = "Iterations Per Hour",
        },
        bar_plots...,
        vline(1.5, yl = 0, yu = ymax, color = "black"),
        vline(1.5 + length(formulations), yl = 0, yu = ymax, color = "black"),

        # System Configuration Comments
        comment(
            1,
            comment_level,
            "System with\\\\160 GB PMM\\\\\\textbf{(lowest cost)}";
            font = font,
        ),
        comment(
            1 + (length(formulations) + 1) / 2,
            comment_level,
            "System with 128 GB PMM\\\\and 32 GB DRAM\\\\\\textbf{(mid cost)}";
            font = font,
        ),
        comment(
            2 + length(formulations),
            comment_level,
            "System with\\\\160 GB DRAM\\\\\\textbf{(highest cost)}";
            font = font,
        )
    )
    push!(tikz, axs)
    push!(plt, tikz)

    pgfsave(file, plt)
    return nothing
end
