# Counters

Record hardware counter metrics for large workloads running in 1LM or 2LM modes.
Used to compare the differences between bandwidths in these two modes and to gain further
insight into the behavior of the 2LM cache.

## Setup

This experiment consists of two files that run in separate Julia processes.
First, start up the performance counter server.
In this directory, perform the following:
```
sudo <path-to-julia> counters.jl
```

In a separate console window, run either `run_1lm.sh` or `run_2lm.sh` as appropriate.
If you desire to run only a subset of these workloads, edit the run scripts.

## Generating LaTeX allocation plots.

Generating the LaTeX plots for the heap allocations is time intensive and kind of a hassle.
To facilitate this running quicker, it is broken into a two step process.

### Generating Timelines

For prettier plots, we generate a timeline of the graph execution by running the graph with profiling counters turned on.
Then, a stripped down copy of the graph is saved along with timing data.
This copy of the graph is the baseline for generating the allocation plots.
These serialized copies will live in the `serialized/` folder.
Note that since the `FunctionData` object contains pointers to nGraph C++ objects, we can't serialize it directly.
Instead, we create a simple `Dict` that we save.
The particular structure of the `Dict` is best understood by just looking at the code in `runner.jl`.

### Generating Plots

Generating each of the heap plots from the timeline takes a while (5 - 10 minutes).
The rendered `pdf`s will be saved to the `figures/` folder.
To speeup up this process, we can use Julia's multiprocessing.
Start Julia with
```
julia -pN
```
where `N` is the number of workers you want to use.
In Julia, run
```julia
juila> @everywhere include("plots.jl")

julia> files = readdir("serialized")

julia> pmap(CounterPlots.heap_plot, joinpath.(pwd(), "serialized", files))
```
This will distribute the plotting to all the workers, taking advantage of the multicore
world we've all come to know and love!

## Rendering Notebook

The summary of the experiment is stored in `summary.jmd`, which is an input file for the
[Weave](https://github.com/JunoLab/Weave.jl) package.
Weave can be used to convert this into a full Jupyter notebook.
In this directory, launch Julia and run the commands:
```julia
using Pkg; Pkg.activate("../../AutoTM")
using Weave
convert_doc("summary.jmd", "summary.ipynb")
```
The notebook `summary.ipynb` should then be runnable.

**NOTE**: Make sure all of the heap plot figures are rendered before trying to run the notebook.

### Two Process Reasoning

By default, all julia I/O tasks, timers etc are multiplexed through a single OS thread.
This includes calls into shared libraries.
Thus, calling into the ngraph library essentiall stops the same Julia process from concurrently reading hardware counters.
Julia provides a hook to get around this with the `@threadcall` macro
(https://docs.julialang.org/en/v1/manual/parallel-computing/#@threadcall-(Experimental)-1), but since we are using [CxxWrap](https://github.com/JuliaInterop/CxxWrap.jl), we don't get this option.
Therefor, two separate processes are used - one monitoring the performance counters and one for managing all of the ngraph stuff.
The two processes communicate through a NamedPipe.
Since we're using two processes, and performance counters need `sudo` to work correctly, we can run the performance counter code under `sudo` and the other code as a normal user.

## Test Command

To test if the counters are working
```
julia --color=yes runner.jl --mode=1lm --workload=test_vgg --counter_type rw queue
```
NOTE: Make sure that `counters.jl` is running.

To test the function saving properties run the command
```
julia --color=yes runner.jl --mode=2lm --workload=test_vgg --save_kerneltimes
```

## Hypothesis

Size of dirty cache must be "X" percent of the DRAM cache.
What is the percent of the cache that can be dirty and still give reasonable performance.
Possibly one quarter is enough.
