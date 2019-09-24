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
            # push!(x, ratio_string(ratio))
            # if as_ratio
            #     push!(y, (cost / all_dram_cost) / perf) 
            # else
            #     push!(y, cost / perf)
            # end
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
