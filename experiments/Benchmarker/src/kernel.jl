# Function for plotting the performance of a specific kernel in order to illustrate
# performance difference between PMM and DRAM
using nGraph
using nGraph.Flux
using nGraph.NNlib

# This is the implementation of the convolution kernel for profiling.
# We do some addition tricks to make sure intermediate tensors exist.
function __kernel(x, w)
    x += ones(eltype(x), size(x))
    w += ones(eltype(x), size(w))
    z = NNlib.conv(x, w; pad = (1, 1, 1, 1))
    return log(z)
end

function profile_kernel()
    # Input data
    x = rand(Float32, 48, 48, 128, 128)
    # Input weights
    w = rand(Float32, 3, 3, 128, 128)

    # Create an Actualizer and do the profiling.
    f = () -> AutoTM.Actualizer(__kernel, x, w)
    kernel_profile(f)
end

# Plot the kernel performance
function plot_kernel_performance(;
        file = "plot.pdf",
        width = "15cm",
        height = "5cm",
    )

    # Step 1: Get the kernel from the cache
    cache = AutoTM.Profiler.CPUKernelCache(AutoTM.Experiments.CPU_CACHE)
    configs = (AutoTM.Utils.DRAM, AutoTM.Utils.PMEM)

    expected_input_sizes = (
        (48, 48, 128, 128),
        (3, 3, 128, 128),
    )

    times = Float64[]
    for i in Iterators.product(configs, configs)
        for o in configs
            config = AutoTM.Utils.IOConfig(wrap(i), wrap(o))
            for (k,v) in cache.cache
                params = k[1]
                io_config = k[2]
                if config == io_config &&
                        params.description == "Convolution" &&
                        params.input_sizes == expected_input_sizes
                    push!(times, v)
                end
            end
        end
    end

    # Normalize times to a multple of the "all dram" case.
    times .= times ./ first(times)

    # Now that we've found the kernel times, we generate the plot!!
    xticks = [
        "DRAM\\\\DRAM\\\\DRAM",
        "DRAM\\\\DRAM\\\\\\textbf{PMM}",
        "DRAM\\\\\\textbf{PMM}\\\\DRAM",
        "DRAM\\\\\\textbf{PMM}\\\\\\textbf{PMM}",

        "\\textbf{PMM}\\\\DRAM\\\\DRAM",
        "\\textbf{PMM}\\\\DRAM\\\\\\textbf{PMM}",
        "\\textbf{PMM}\\\\\\textbf{PMM}\\\\DRAM",
        "\\textbf{PMM}\\\\\\textbf{PMM}\\\\\\textbf{PMM}",
    ]

    plot = Plot(Coordinates((i, t) for (i,t) in enumerate(times)))

    plt = TikzDocument()
        push!(plt, """
    \\pgfplotsset{
        width=$width,
        height=$height
    }
    """)

    tikz = TikzPicture()
        axs = @pgf Axis(
        {
            ybar,
            enlarge_x_limits=0.10,
            nodes_near_coords_align={vertical},
            ylabel="Performance relative\\\\to all IO in DRAM",
            ymajorgrids,
            ymin=0,
            ylabel_style={
                align = "center",
            },

            xtick="data",
            xticklabels = xticks,
            xticklabel_style={
                align = "center",
            },
            yticklabel_style={
                "/pgf/number format/fixed",
                "/pgf/number format/precision=5",
            },
            bar_width="20pt",
            clip=false,
        },
        plot,
        """
        \\node[align=right] at (axis cs: -0.1, -0.78) {
        \\textbf{Data In:}\\\\
        \\textbf{Weight:}\\\\
        \\textbf{Data Out:}\\\\
        };
        """,
    )

    push!(tikz, axs)
    push!(plt, tikz)
    pgfsave(file, plt)
    return plt
end

