# Benchmarks

## Running Plot Scripts

If you want to generate all plots and have the expected data in the correct place, the
`plots.jmd` file can be turned into a Jupyter notebook using [Weave](https://github.com/JunoLab/Weave.jl)

Run the command
```julia
using Pkg; Pkg.add("Weave")
using Weave
notebook("plots.jmd")
```
A notebook titled "plots.ipynb" will appear which can be executed to generate the plots.

After running the nodebooks, generated figures will be in the `figures` subdir.

!!! note

    You will have to have a LaTeX distribution installed for figure generation to work correctly. 

## Plots

Below is a summary of the plotting functions.

* `pgf_movement_plot`: Plot the total amount of data moved between DRAM and PMM with PMM/DRAM
    ratio on the x-axis.

* `pgf_io_plot`: Plot of the percent of kernel input and output data is in DRAM. PMM/DRAM
    ratio is given on the x-axis.

* `pgf_plot_performance`: Plot the detailed runtime performance across PMM/DRAM ratios.

* `pgf_speedup`: Plot the speedup of AutoTM over all PMM across various PMM/DRAM ratios.

* `pgf_large`: Plot the speedup of AutoTM over 2LM for the large workloads.

* `pgf_gpu_performance_plot`: Show the performance of the GPU implementation of AutoTM.

* `plot_front`: Plot the front figure for AutoTM.

* `pgf_error_plot`: Generate the error plots between profiled runtime and actual runtime.

* `pgf_price_performance`: Plot of the price/performance of Optane.
