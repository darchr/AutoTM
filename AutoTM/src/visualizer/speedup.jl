# Plot ratios of PMEM to DRAM on the x-axis.
function pgf_speedup(fns, ratios::Vector{<:Rational}, cache;
        file = "plot.tex",
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
    return nothing
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
            title = titlename(f),
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
