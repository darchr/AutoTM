# AutoTM Artifact Workflow

```@contents
Pages = ["benchmarker.md"]
Depth = 3
```

## PMM

## GPU

### Preparation
Navigate to the Benchmarker directory
```sh
cd $AUTOTM_HOME/experiments/Benchmarker
```

Run a new Julia session
```sh
julia --project
```
In the Julia REPL, make sure all dependencies are installed
```julia
julia> ]

(Benchmarker) pkg> instantiate
```

### Profiling

Because of memory overheads, the GPU experiments are split into two parts.
The first part involves generating the kernel profile information.
The second part is the actual running of the experiments themselves.

To generate the kernel profile data, perform the following sequence of commands in the `Benchmarker` directory
```julia
using Benchmarker, AutoTM

Benchmarker.gpu_profile()
```
when the system finishes profiling, exit the Julia session.

### Running Benchmarks

In a new Julia session, run the benchmarks with
```julia
using Benchmarker, AutoTM

Benchmarker.gpu_benchmarks()
```

!!! note
    
    There are some default variables set for the amount of GPU DRAM and for the overhead of the ngraph/CUDA runtimes.
    These are set to 11 GB and 1 GB respectively for a RTX 2080Ti.
    With a different GPU/CUDA version, these will need to be changed.
    For example, if your GPU has 6 GB of memory, these values may be set using
    ```julia
    using Benchmarker, AutoTM
    
    Benchmarker.GPU_MAX_MEMORY[] = 6_000_000
    Benchmarker.GPU_MEMORY_OVERHEAD[] = 1_000_000
    ```
    Memory overhead can be queried using `nvidia-smi`

### Generating Plots

Following benchmark runs, the GPU performance plot (Figure 12) are simply generated using

```julia
Benchmarker.gpu_performance_plot()
```

