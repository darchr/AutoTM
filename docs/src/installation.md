# Installation

## Software Requirements

AutoTM was developed and tested using Ubuntu 18.04 using [Julia 1.2](https://julialang.org/downloads/oldreleases.html).
We expect it to work on similar operating systems and non-breaking future versions of Julia.

The following are required to build the ngraph dependency:
```
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

### PMM System

If you are running on a system equipped with Intel Optane DC PMMs, the following packages are required
```
numactl \
ipmctl  \
ndctl   
```

### GPU System

If you are using a system with an NVidia GPU, you will additionally need CUDA 10.1/10.2 and cuDNN 7.6.

## Getting Code

Clone the repository with
```sh
git clone --recursive https://github.com/darchr/AutoTM
export AUTOTM_HOME=$(pwd)/AutoTM
```

## Setup

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
    
## Building

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

## Using the Gurobi ILP solver (optional)

The results in the AutoTM paper use [Gurobi](https://www.gurobi.com) for the ILP solver.
However, Gurobi requires a license to run.
Free trial and academic licenses are available from the Gurobi website: https://www.gurobi.com

If using Gurobi, please obtain a license and install the software according the instructions on the website.

Then, when building the project, make sure to run
```julia
julia> ENV["GUROBI_HOME"] = "path/to/gurobi"
```
in Julia before executing the build step above.

!!! note

    Using the Gurobi ILP solver is optional.
    If not selected during the setup step, an open-source solver [Cbc](https://projects.coin-or.org/Cbc) will be used.

