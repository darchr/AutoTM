# AutoTM

## Sub Projects

| Repo | Link   | Description |
|------|--------|-------------|
| ngraph (fork)       | https://github.com/darchr/ngraph/tree/mh/pmem  | Customized fork of ngraph source code |
| nGraph.jl           | https://github.com/hildebrandmw/nGraph.jl      | Julia frontend for nGraph             |
| PersistentArrays.jl | https://github.com/darchr/PersistentArrays.jl  | NVDIMM backed arrays |
| SystemSnoop.jl      | https://github.com/hildebrandmw/SystemSnoop.jl | Base System monitoring API |
| PCM.jl              | https://github.com/hildebrandmw/PCM.jl         | Wrapper for Intel [pcm](https://github.com/opcm/pcm)  |
| pcm                 | https://github.com/hildebrandmw/pcm | Customized fork of Intel pcm |
| MaxLFSR.jl          | https://github.com/hildebrandmw/MaxLFSR.jl | Maximum length Linear Feedback Shift Registers |

# Requirements

Ubuntu 18.04 LTS

Install required packages with
```sh
build-essential \
cmake \
clang-6.0 \
clang-format-6.0 \
git \
curl \
zlib1g \
zlib1g-dev \
libtinfo-dev \
unzip \
autoconf \
automake \
libtool
```

# Installation

Clone the repository with
```sh
git clone --recursive https://github.com/darchr/AutoTM
export AUTOTM_HOME=$(pwd)/AutoTM
```

### Setup

A simple setup needs to be performed to indicate how the project will be used.
To enter the setup, run
```sh
cd $AUTOTM_HOME
julia --color=yes setup.jl
```
The following selections can be made - choose which are appropriate for your system:
* Use NVDIMMs in 1LM (requires a Cascade Lake system with Optane DC NVDIMMs)
* Use of a GPU (requires CUDA 10.1 or CUDA 10.2)
* Use Gurobi as the ILP solver (requires a Gurobi license (see below)).
    If Gurobi is not selected, the open source Cbc solver will be used.
    Please note that the original experiments were run with Gurobi.
    
### Building

Launch Julia from the AutoTM project
```sh
cd $AUTOTM_HOME/AutoTM
julia --project
```

In the Julia REPL, press `]` to switch to package (pkg) mode and run following commands:
```julia
julia> ]
(AutoTM) pkg> instantiate
(AutoTM) pkg> build -v
```
This will trigger the build process for our custom version of ngraph.
Passing the `-v` command to `build` will helpfully display any errors that occur during the build process.

### Using the Gurobi ILP solver (optional)

The results in the AutoTM paper use [Gurobi](https://www.gurobi.com) for the ILP solver.
However, Gurobi requires a license to run.
Free trial and academic licenses are available from the Gurobi website: https://www.gurobi.com

If using Gurobi, please obtain a license and install the software according the instructions on the website.

Then, when building the project, make sure to run
```julia
julia> ENV["GUROBI_HOME"] = "path/to/gurobi"
```
in Julia before executing the build step above.

**NOTE**: Using the Gurobi ILP solver is optional.
If not selected during the setup step, an open-source solver [Cbc](https://projects.coin-or.org/Cbc) will be used.

## Running Experiments

```julia
# Running this for the first time triggers precompilation which make take a couple minutes
julia> using AutoTM

julia> exe, _ = AutoTM.Optimizer.factory(
    AutoTM.Backend("GPU"),
    AutoTM.Experiments.test_vgg(),
    AutoTM.Optimizer.Asynchronous(AutoTM.Experiments.GPU_ADJUSTED_MEMORY);
    cache = AutoTM.Profiler.GPUKernelCache(AutoTM.Experiments.GPU_CACHE)
);

# The model can be run by simply executing
julia> exe()
Tensor View
0-dimensional CuArrays.CuArray{Float32,0}:
19.634962
```

## Saving and Loading Experimental Data

All of the code for AutoTM, including code for experiments, is version controlled in various Git repositories.
However, results from experiments, which can include serialized Julia data structures as well as figures, are not version controlled due to size limitations.
Raw data for the `Benchmarker` suite of experiments exceeds 100 MB.
There is a mechanism to save and load all experimental data.

### Saving

To save experimental data, navigate to the `scripts/` folder and run the command
```
julia saver.jl save
```
This will gather all of the relevant experimental data, as well as the kernel profile cache and bundle it into the tarball `autotm_backup.tar.gz`.

### Loading
To reverse this process, run
```
julia saver.jl load --tarball <path/to/tarball> [--force]
```
This will unpack the tarball and put all the contents back where they came from.
The `--force` flag will remove existing directories and replace them with the tarball data.
