#####
##### Plotting Utility Functions
#####

rectangle(x, y, w, h) = (x .+ [0, w, w, 0]), (y .+ [0, 0, h, h])
getname(v, s::Symbol) = getindex.(v, s)

load_save_files(f, formulations::String) = first(load_save_files(f, (formulations,)))
function load_save_files(f, formulations)
    savefiles = [joinpath(savedir(f), join((name(f), i), "_") * ".jls") for i in formulations]
    data = deserialize.(savefiles)
    for d in data
        sort!(d.runs; rev = true, by = x -> get(x, :dram_limit, 0))
    end

    return data
end

# For drawing a vertical asymptote on a graph
vasymptote() = """
\\pgfplotsset{vasymptote/.style={
    before end axis/.append code={
        \\draw[densely dashed] ({rel axis cs:0,0} -| {axis cs:#1,0})
        -- ({rel axis cs:0,1} -| {axis cs:#1,0});
}
}}
"""

hasymptote() = """
\\pgfplotsset{hasymptote/.style={
    before end axis/.append code={
        \\draw[densely dashed] ({rel axis cs:0,0} -| {axis cs:0,#1})
        -- ({rel axis cs:1,0} -| {axis cs:0,#1});
}
}}
"""

hline(y; xl = 0, xu = 1, color = "red") = """
\\draw[$color, densely dashed, ultra thick]
    ({axis cs:$xl,$y} -| {rel axis cs:0,0}) --
    ({axis cs:$xu,$y} -| {rel axis cs:1,0});
""" |> rm_newlines

vline(x; yl = 0, yu = 1, color = "red") = """
\\draw[$color, sharp plot] ($x, $yl) -- ($x, $yu);
""" |> rm_newlines

comment(x, y, str; kw...) = @pgf(["\\node[align = center$(format(kw))] at ", Coordinate(x, y), "{$str};"])
format(kw) = ", " * join(["$a = $b" for (a,b) in kw], ", ")

rm_newlines(str) = join(split(str, "\n"))

# Node - must sort data before hand
# using load_save_files does this automatically
get_dram_performance(data) = minimum(get_dram_performance.(data))
get_dram_performance(data::NamedTuple) = first(getname(data.runs, :actual_runtime))

get_pmm_performance(data) = maximum(get_pmm_performance.(data))
get_pmm_performance(data::NamedTuple) = last(getname(data.runs, :actual_runtime))

function findabsmin(f, x)
    _, ind = findmin(abs.(f.(x)))
    return ind
end

#####
##### Statistics on what AutoTM has done.
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
    return plot
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
    return plot
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
            ylabel = "Slow Down Relative\\\\to all DRAM",
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
    return plot
end

#####
##### Speedup over all PMM
#####

# Plot ratios of PMEM to DRAM on the x-axis.
function pgf_speedup(fns, ratios::Vector{<:Rational}, cache;
        file = joinpath(FIGDIR, "speedup.pdf"),
        formulations = ("numa", "static", "synchronous")
    )

    doc = TikzDocument()
    push!(doc, hasymptote())

    count = 1
    axes = map(fns) do f
        axs = _speedup(f, ratios, cache, formulations;
            legend_entries = (count == 1),
            top = (count == 1),
            bot = (count == length(fns)),
            left_label = in(count, (1, 3)),
            bottom_label = in(count, (3, 4)),
        )
        count += 1
        return axs
    end

    gp = @pgf GroupPlot(
        {
            height = "4cm",
            width = "6cm",
            group_style = {
                group_size = "2 by 2",
                vertical_sep = "1.5cm",
            },
        },
        axes...
    )

    pic = TikzPicture(gp)

    # Post processing for legends and stuff
    push!(pic, raw"\path (top)--(bot) coordinate[midway] (group center);")
    push!(pic, raw"\node[below=4.5cm] at(group center -| current bounding box.south) {\pgfplotslegendfromname{legend}};")

    push!(doc, pic)
    pgfsave(file, doc)
    return doc
end

const _speedup_formulation_lut = Dict(
    "static" => "static-AutoTM",
    "synchronous" => "sync-AutoTM",
)

# Inner plot for each `f`
function _speedup(f, ratios, cache, formulations;
        legend_entries = false,
        top = false,
        bot = false,
        left_label = false,
        bottom_label = false,
    )
    paths = canonical_path.(Ref(f), formulations, Ref(cache), Ref(nGraph.Backend("CPU")))
    data = deserialize.(paths)

    pmm_performance = get_pmm_performance(data)
    dram_performance = get_dram_performance(data)

    plots = []
    for (datum, formulation) in zip(data, formulations)
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

            perf = pmm_performance / datum.runs[ind][:actual_runtime]
            push!(x, ratio_string(ratio))
            push!(y, perf)
        end

        # Emit the plot for this series.
        push!(plots,
            @pgf(PlotInc(
                Coordinates(x, y),
            ))
        )

        if legend_entries
            push!(
                plots,
                @pgf(LegendEntry(get(_speedup_formulation_lut, formulation, formulation)))
            )
        end
    end

    symbolic_coords = ratio_string.(ratios)
    dline = pmm_performance / dram_performance

    options = []
    if legend_entries
        push!(options, "legend to name = legend")
    end

    if left_label
        push!(options, "ylabel = Speedup over all PMM")
    end

    if bottom_label
        push!(options, "xlabel = PMM to DRAM Ratio")
    end

    axs = @pgf Axis(
        {
            title = AutoTM.Experiments.titlename(f),
            enlarge_x_limits=0.30,
            ybar,
            ymin=0,
            symbolic_x_coords = symbolic_coords,
            nodes_near_coords_align={vertical},
            ymajorgrids,
            xtick="data",
            ytick = 1:(ceil(Int, pmm_performance / dram_performance)+1),
            ymax = ceil(Int, pmm_performance / dram_performance),
            options...,
            legend_style = {
                legend_columns = -1,
            },
        },
        plots...,
        # Line for All DRAM performance
        hline(dline;
              xl = first(symbolic_coords),
              xu = last(symbolic_coords)
             ),
    )

    if legend_entries
        push!(axs, raw"\addlegendimage{line legend, red, densely dashed, ultra thick}")
        push!(axs, @pgf(LegendEntry("All DRAM")))
    end

    if top
        push!(axs, "\\coordinate (top) at (rel axis cs:0,1);")
    end

    if bot
        push!(axs, "\\coordinate (bot) at (rel axis cs:1,0);")
    end

    return axs
end

#####
##### Performance of AutoTM for the large workloads.
#####

function pgf_large_performance(fns, cache, cache_2lm;
        file = joinpath(FIGDIR, "large_performance.pdf"),
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

        push!(coords, "$(AutoTM.Experiments.titlename(f)) ($(f.batchsize))")
    end

    @show coords

    for formulation in formulations
        x = []
        y = []

        for (i, f) in enumerate(fns)
            data = deserialize(canonical_path(f, formulation, cache, backend))
            speedup = baseline_runtime[f] / minimum(getname(data.runs, :actual_runtime))
            push!(x, "$(AutoTM.Experiments.titlename(f)) ($(f.batchsize))")
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
            legend_style =
            {
                 at = Coordinate(0.05, 1.05),
                 anchor = "south west",
                 legend_columns = -1,
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
    return plt
end

#####
##### Plots for GPU code
#####

function pgf_gpu_performance_plot(
        funcs, cache;
        file = joinpath(FIGDIR, "gpu_performance.pdf"),
        formulations = ("synchronous", "asynchronous")
    )

    backend = nGraph.GPU

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
    for title in unique(AutoTM.Experiments.titlename.(funcs))
        # Get the start and stop indices for this title
        start = findfirst(x -> AutoTM.Experiments.titlename(x) == title, funcs)
        stop = findlast(x -> AutoTM.Experiments.titlename(x) == title, funcs)
        center = (start + stop) / 2

        push!(annotations, comment(center, network_offset, title))
    end

    # Generate data for the formulations
    for formulation in formulations
        y = []
        x = []

        for (i, f) in enumerate(funcs)
            @show name(f)
            data = deserialize(canonical_path(f, formulation, cache, backend))
            speedup = baseline_runtime[f] / minimum(getname(data.runs, :actual_runtime))

            println("Runtime - $formulation - $f:", minimum(getname(data.runs, :actual_runtime)))

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
    return plt
end

#####
##### The front plot.
#####

iterations_per_hour(runtime) = 3600 / (runtime)

const LEGEND_LUT = Dict(
    "synchronous" => "\\textbf{AutoTM}",
    "numa" => "NUMA",
)
_legend(x) = get(LEGEND_LUT, x, x)

function plot_front(f, ratio::Rational, cache;
        formulations = ("numa", "synchronous"),
        file = joinpath(FIGDIR, "front.pdf")
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
    return plt
end

#####
##### Error from profiling
#####

function pgf_error_plot(fns, ratios, caches;
        file = joinpath(FIGDIR, "error.pdf"),
        formulations = ("static", "synchronous"),
        # Older versions of the ILP formulation measured time in micro-seconds instead of 
        # 100ths of a second.
        #
        # This flag adjusts that.
        legacy = false,
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

                    if legacy
                        predicted_runtime = data.runs[ind][:predicted_runtime] / 1E6
                    else
                        predicted_runtime = data.runs[ind][:predicted_runtime] / 1E2
                    end

                    push!(x, ratio_string(ratio))
                    push!(y, 100 * (predicted_runtime - actual_runtime) / actual_runtime)
                end

                append!(plots, [
                    @pgf(PlotInc(Coordinates(x, y)))
                    @pgf(LegendEntry("$(AutoTM.Experiments.titlename(f)) - $formulation"))
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

    return plt
end

#####
##### Price Performance across a range of DRAM/PMM costs.
#####

function pgf_price_performance(
        pairs::Vector{<:Pair},
        ratios::Vector{<:Rational},
        cache;

        base_cost = 0,
        dram_cost_per_gb = 0,
        pmm_cost_per_gb = 0,
        as_ratio = false,
        # Can be `nothing` if no file to be emitted
        file = "plot.tex",
        width = 7,
        height = 4,
    )

    # Initialize a vector of PGFPlotsX structs that will be used to generate the actual
    # plot.
    plots = []
    backend = nGraph.Backend("CPU")

    for (f, formulation) in pairs
        #data = load_save_files(f, formulation)
        data = deserialize(canonical_path(f, formulation, cache, backend))
        dram_performance = get_dram_performance(data)

        # Compute the dram cost
        all_dram_cost = base_cost +
            (dram_cost_per_gb * first(data.runs)[:dram_alloc_size] / 1E9) +
            (pmm_cost_per_gb * first(data.runs)[:pmem_alloc_size] / 1E9)

        @show first(data.runs)[:dram_alloc_size] / 1E9
        @show all_dram_cost

        # This x and y data point
        x = []
        y = []

        for (i, ratio) in enumerate(ratios)
            ind = findabsmin(x -> compare_ratio(getratio(x), ratio), data.runs)
            perf = dram_performance / data.runs[ind][:actual_runtime]
            pmm_amount = data.runs[ind][:pmem_alloc_size]
            dram_amount = data.runs[ind][:dram_alloc_size]
            cost = base_cost +
                (dram_cost_per_gb * dram_amount / 1E9) +
                (pmm_cost_per_gb * pmm_amount / 1E9)

            push!(x, cost / all_dram_cost)
            push!(y, perf)
        end

        # Emit the plot for this series.
        append!(plots, [
            @pgf(PlotInc(
                Coordinates(x, y),
            )),
            @pgf(LegendEntry(replace(AutoTM.Experiments.titlename(f), "_" => " "))),
        ])
    end

    plt = TikzDocument()
    push!(plt, """
    \\pgfplotsset{width=$(width)cm,height=$(height)cm}
    """)

    symbolic_coords = ratio_string.(ratios)

    push!(plots, @pgf(PlotInc(Coordinates([0, 1], [0, 1]))))

    # Bar axis
    tikz = TikzPicture()
    axs = @pgf Axis(
        {
            #ybar,
            #bar_width = "5pt",
            legend_style =
            {
                 at = Coordinate(1.05, 1.05),
                 anchor = "north west",
                 legend_columns = 1,
            },
            ymin = 0,
            #symbolic_x_coords = symbolic_coords,
            nodes_near_coords_align={vertical},
            ymajorgrids,
            xmajorgrids,
            ylabel_style={
                align = "center",
            },
            title = "Base Cost: $base_cost. DRAM Cost: $dram_cost_per_gb. PMM Cost: $pmm_cost_per_gb",
            #ytick = [0,1],
            # Lables
            xlabel = "Cost Ratio to all DRAM",
            ylabel = "Perforamnce Ratio to all DRAM",

        },
        plots...,
    )
    push!(tikz, axs)
    push!(plt, tikz)

    isnothing(file) || pgfsave(file, plt)
    return plt
end

#####
##### The cost-performance plot from the ASPLOS 20 submission.
#####

"""
`pairs`: Vector of Pairs, first element is a model, second element is a formulation string.
"""
function pgf_cost(
        pairs::Vector{<:Pair},
        ratios::Vector{<:Rational},
        cache;
        # Can either define cost_ratio or the costs of PMM and DRAM separately
        cost_ratio = 2.1,
        # location of output file - if `nothing`, no file will be emitted
        file = joinpath(FIGDIR, "costs.pdf"),
    )

    backend = nGraph.Backend("CPU")
    plots = []
    for (f, formulation) in pairs
        data = deserialize(canonical_path(f, formulation, cache, backend))
        dram_performance = get_dram_performance(data)

        # This x and y data point
        x = []
        y = []

        for ratio in ratios
            ind = findabsmin(x -> compare_ratio(getratio(x), ratio), data.runs)
            perf = dram_performance / data.runs[ind][:actual_runtime]

            push!(x, ratio_string(ratio))
            push!(y, perf)
        end

        # Emit the plot for this series.
        append!(plots, [
            @pgf(PlotInc(
                Coordinates(x, y),
            )),
            @pgf(LegendEntry(replace(AutoTM.Experiments.titlename(f), "_" => " "))),
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
            "axis_y_line*=left",
            "scale_only_axis",
            ybar,
            bar_width = "5pt",
            legend_style =
            {
                 at = Coordinate(0.05, 1.05),
                 anchor = "south west",
                 legend_columns = 2,
            },
            ymin = 0,
            ymax = 1,
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
            ylabel = "Performance Relative to all DRAM",

        },
        plots...,
    )
    push!(tikz, axs)

    # Cost Axis
    y_cost = map(ratios) do ratio
        # PMM
        num = ratio.num

        # DRAM
        den = ratio.den

        # Total approximately sums to 1
        return (num / cost_ratio + den) / (num + den)
    end

    axs = @pgf Axis(
        {
            "axis_y_line*=right",
            "scale_only_axis",
            axis_x_line = "none",
            only_marks,
            ymin = 0,
            ymax = 1,
            symbolic_x_coords = symbolic_coords,
            ylabel = "Memory cost\\\\relative to all DRAM"
        },
        PlotInc(
            {
                mark = "text",
                text_mark = raw"\$",
                mark_options = {
                    color = "black",
                    scale = 1.5,
                },
            },
            Coordinates(symbolic_coords, y_cost),
        ),
    )
    push!(tikz, axs)
    push!(plt, tikz)

    !isnothing(file) && pgfsave(file, plt)
    return plt
end
