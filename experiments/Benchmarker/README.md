# Benchmarks

## Startup

This benchmark relies on the stacking property of Julia environments.
We have a base environment from AutoTM, and add extra plotting capabilities on top of it.
To correctly use this subpacket, start Julia and run the command
```julia
julia> include("init.jl")
```
This will correctly stack environments on the `LOAD_PATH`.
You can then run
```
julia> using Benchmarker
```
successfully.

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
