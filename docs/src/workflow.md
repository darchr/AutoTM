# AutoTM Artifact Workflow

This section outlines how to run the experiments performed in the AutoTM paper and generate Figures 7 to 12 from the paper.
The code to run these experiments lives in `$AUTOTM_HOME/experiments/Benchmarker`.
Unless otherwise specified, all commands given below should be executed from this directory.
Julia should be started with `julia --project`.

```@contents
Pages = ["benchmarker.md"]
Depth = 3
```

## PMM - Configuring 1LM and 2LM

Servers with Intel Optane DC can be configured to run in either 1LM/AppDirect mode, where reads and writes to PMM are managed manually, or 2LM/Memory Mode where PMM is accessed as main memory with DRAM as a transparent cache.

Most of the AutoTM code expects to run in 1LM mode with PMM mounted to `/mnt/public`.
Scripts are provided in the `$AUTOTM_HOME/scripts` directory to aid in switching modes.

### Switching to 1LM

Reboot the system and select 1LM in the BIOS.
After reboot, navigate to `$AUTOTM_HOME/scripts` and run
```sh
sudo ./change_1lm.sh
```
Reboot the system again.
After the system comes online again, navigate back to `$AUTOTM_HOME/scripts` and run
```sh
sudo ./setup_1lm.sh
```

!!! note

    The script `setup_1lm.sh` will destroy all data in PMM namespace 1.0.
    **DO NOT** run this script if there is any data on there that must be preserved.

The setup script will create a new file system on the NVDIMMs on Socket 1 and perform a direct-access filesystem mount to `/mnt`.

### Switching to 2LM

Reboot the system and select 2LM in the BIOS.
After reboot, navigate to `$AUTOTM_HOME/scripts` and run
```sh
sudo ./change_2lm.sh
```
Reboot the system again.
That is all.

## PMM - Conventional Benchmarks

Make sure the system is in AppDirect mode and that `setup_1lm.sh` has been executed.

### Kernel Profiling

Kernel timing profiling must happen separately before the actual execution of benchmarks due to memory fragmentation.

To perform kernel profiling, run
```julia
using Benchmarker, AutoTM
Benchmarker.kernel_profile(Benchmarker.conventional_functions())
```
Kernel profiling for all networks can take hours.
Grab a cup of coffee and let AutoTM do its thing.

The serialized data structure for the cached kernel profiles lives in `$AUTOTM_HOME/data/caches`.

### Running Benchmarks

Reboot the system before running these benchmarks.
Ensure the system is under light load for best results.

```julia
using Benchmarker, AutoTM

optimizers = [
    AutoTM.Optimizer.Static,
    AutoTM.Optimizer.Synchronous,
    AutoTM.Optimizer.Numa
]

ratios = Benchmarker.common_ratios()

for fn in Benchmarker.conventional_functions()
    Benchmarker.run_conventional(fn, optimizers, ratios)
end
```

Results for these runs will be stored to `$AUTOTM_HOME/experiments/Benchmarker/data/cpu`

### Generating Plots

To generate Figures 7, 9, and 11 - run the following
```julia
using Benchmarker

# Figure 7
Benchmarker.plot_speedup()

# Figure 9
Benchmarker.plot_costs()

# Figure 11
Benchmarker.plot_conventional_error()
```

### Test Run

For verification purposes, a small Vgg19 network is included.
```
using Benchmarker, AutoTM
Benchmarker.kernel_profile(Benchmarker.test_vgg())
Benchmarker.run_conventional(
    Benchmarker.test_vgg(),
    [AutoTM.Optimizer.Static, AutoTM.Optimizer.Synchronous, AutoTM.Optimizer.Numa],
    Benchmarker.common_ratios(),
)

# Generate Plots
Benchmarker.plot_speedup(
    models = [Benchmarker.test_vgg()],
)

Benchmarker.plot_conventional_error(
    models = [Benchmarker.test_vgg()],
)

Benchmarker.plot_costs(
    pairs = [Benchmarker.test_vgg() => "synchronous"],
)
```

## PMM - Inception Case Study

This experiment explores the sensitivity of the ILP formulation to PMM/DRAM ratios.
Make sure the kernels are profiled prior to performing this experiment.

### Running the Experiment

The Inception Case study simply involves running the `conventional_inception()` workload for a large number of PMM to DRAM ratios.
```julia
using Benchmarker
Benchmarker.inception_case_study()
```

### Generating Plots

To generate Figure 10a, 10b, and 10c, run
```julia
using Benchmarker
Benchmarker.inception_case_study_plots()
```

## PMM - Large Networks

This experiment compares AutoTM with the hardware managed 2LM.
The workloads used for this experiment all used on the order of 650 GB of memory and so far exceed the size of local DRAM.

### Kernel Profiling

As with the conventional workloads, kernel profiling must be performed.
The command given below will perform all profiling.
Be warned that because of the large number of unique kernels in DenseNet, profiling can take about a day.
Thus, you may want to just run a subset of the workloads.

```julia
using Benchmarker

workloads = [
    Benchmarker.large_vgg(),
    Benchmarker.large_inception(),
    Benchmarker.large_resnet(),
    Benchmarker.large_densenet()
]

Benchmarker.kernel_profile(workloads)
```

### AutoTM Data

Due to the large size of these workloads, the system should be rebooted between each run to minimize memory fragmentation.
It's not absolutely necessary, but can help with consistency.

```julia
using Benchmarker, AutoTM

### Run each of the large workloads

# Vgg
Benchmarker.run_large(Benchmarker.large_vgg(), AutoTM.Optimizer.Static)
Benchmarker.run_large(Benchmarker.large_vgg(), AutoTM.Optimizer.Synchronous)

# Inception
Benchmarker.run_large(Benchmarker.large_inception(), AutoTM.Optimizer.Static)
Benchmarker.run_large(Benchmarker.large_inception(), AutoTM.Optimizer.Synchronous)

# Resnet
Benchmarker.run_large(Benchmarker.large_resnet(), AutoTM.Optimizer.Static)
Benchmarker.run_large(Benchmarker.large_resnet(), AutoTM.Optimizer.Synchronous)

# DenseNet
Benchmarker.run_large(Benchmarker.large_densenet(), AutoTM.Optimizer.Static)
Benchmarker.run_large(Benchmarker.large_densenet(), AutoTM.Optimizer.Synchronous)
```

### 2LM Data

Switch over the system to 2LM using the process outlined above.
Once the system is in 2LM, run the following commands


```julia
using Benchmarker

Benchmarker.run_2lm(Benchmarker.large_vgg())
Benchmarker.run_2lm(Benchmarker.large_inception())
Benchmarker.run_2lm(Benchmarker.large_resnet())
Benchmarker.run_2lm(Benchmarker.large_densenet())
```

### Generating Plots

This generates Figure 8.

```julia
using Benchmarker

Benchmarker.plot_large()
```

## GPU

### Preparation

To allow data to be moved to the host system, CUDA needs pinned memory.
Make sure to run `ulimit -l` to allow unlimited pinned host memory before running.

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

Julia must be restarted between each benchmark.
This is because while ngraph is responsible for one large allocation for intermediate data, input and outputs tensors on the Julia side are managed by [CuArrays](https://github.com/JuliaGPU/CuArrays.jl)
These two allocation sources generally confuse each other across multiple runs, so the most consistent way to get results is to restart Julia.

A script has been provided in the `Benchmarker` directory.
To run it, execute
```
julia --color=yes gpu_script.jl
```

!!! note

    There are some default variables set for the amount of GPU DRAM and for the overhead of the ngraph/CUDA runtimes.
    These are set to 11 GB and 1 GB respectively for a RTX 2080Ti.
    With a different GPU/CUDA version, these will need to be changed.
    For example, if your GPU has 6 GB of memory, these values may be set using
    ```julia
    using Benchmarker, AutoTM

    Benchmarker.GPU_MAX_MEMORY[] = 6_000_000_000
    Benchmarker.GPU_MEMORY_OVERHEAD[] = 1_000_000_000
    ```
    Memory overhead can be queried using `nvidia-smi`

### Generating Plots

Following benchmark runs, the GPU performance plot (Figure 12) are simply generated using

```julia
Benchmarker.gpu_performance_plot()
```
