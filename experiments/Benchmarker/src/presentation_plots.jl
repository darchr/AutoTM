# Inception
function presentation_inception()
    f = conventional_inception()
    cache = AutoTM.Experiments.CPU_CACHE
    suffix = "study"

    # Custom ylabel
    ylabel = "Slowdown\\\\\\textbf{Lower is Better}"
    xlabel = "Dram Limit (GB)\\\\\\textbf{Lower is Better}"

    # Performance line graph
    only_dram = pgf_plot_performance(f, cache, suffix;
        file = joinpath(FIGDIR, "presentation_inception_dram.pdf"),
        formulations = ("synchronous",),
        filter_function = (x, y) -> ([first(x)], [first(y)]),
        ylabel = ylabel,
        xlabel = xlabel,
    )

    only_pmem = pgf_plot_performance(f, cache, suffix;
        file = joinpath(FIGDIR, "presentation_inception_pmm.pdf"),
        formulations = ("synchronous",),
        filter_function = (x, y) -> ([last(x)], [last(y)]),
        ylabel = ylabel,
        xlabel = xlabel,
    )

    all = pgf_plot_performance(f, cache, suffix;
        file = joinpath(FIGDIR, "presentation_inception_all.pdf"),
        formulations = ("synchronous",),
        ylabel = ylabel,
        xlabel = xlabel,
    )

    return (
        only_dram = only_dram,
        only_pmem = only_pmem,
        all = all,
    )
end

function presentation_2lm()
    fns = (large_vgg(), large_inception(), large_resnet(), large_densenet())
    cache = AutoTM.Experiments.CPU_CACHE
    cache_2lm = "nocache"
    formulations = ("static", "synchronous")

    pgf_large_performance(
        fns,
        cache,
        cache_2lm;
        formulations = formulations,
        ylabel = "Speedup over 2LM\\\\\\textbf{Higher is Better}",
        file = joinpath(FIGDIR, "presentation_large_performance.pdf")
    )
end

