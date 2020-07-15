# Counters2

Yet another setup for taking performance counter measurements of AutoTM, this time using more mature tools than the first.

## Getting Cached Kernels

Kernel profiling takes quite a while.
To use a cache of already profiled kernels for Optane DC systems, grab the source dump from: https://github.com/darchr/AutoTM/releases/tag/v0.1.3.
A direct link to the download is: https://github.com/darchr/AutoTM/releases/download/v0.1.3/autotm.tar.gz

Untar the tarball and enter the `AutoTM` directory.
Inside, there is yet another tarball `autotm_data.tar.gz`.
Uncompress this and it will make a directory `autotm_backup`.
To put the kernel cache in a place where AutoTM can find it, run the following commands:
```
mkdir data
cp -r autotm_backup/top_data/caches data
```
Now, AutoTM will skip the majority of profiling when trying to rerun experiments.

## Gurobi

If multiple Gurobi license files are installed, then the default might not be the correct file.
In that case, set the `GRB_LICENSE_FILE` environment variable to point to the correct place, either through
```
export GRB_LICENSE_FILE = "/home/mark/gurobi.lic"
```
or in Julia
```julia
juila> ENV["GRB_LICENSE_FILE"]= "/home/mark/gurobi.lic"
```

## Data Collection

### 2LM

Note: I reset the computer between each run.
This is probably not strictly necessary, but helps yield consistent results.

**Data Collection Daemon**:
We use the `MattDaemon` package to gather performance counter data in a separate process so we don't have to run Gurobi as a root user.
Navigate to the `Counters2` directory.
Launch julia 1.3 under `sudo` with
```
sudo <path/to/julia-1.3> --project
```
Inside the Julia REPL
```julia
using MattDaemon
MattDaemon.runserver(2000)
```

**Run Commands**:
Launch Julia with `numactl --physcpubind=24-47 --membind=1 <path/to/julia-1.3> --project`.
```
using Counters2, AutoTM, nGraph
# DenseNet
Counters2.experiment2lm(AutoTM.Experiments.large_densenet())

# Inception
Counters2.experiment2lm(AutoTM.Experiments.large_inception())

# Resnet
Counters2.experiment2lm(AutoTM.Experiments.large_resnet())
```

### 1LM

Setup the data collection Daemon as before.
The commands to run are
Launch Julia with `numactl --physcpubind=24-47 --membind=1 <path/to/julia-1.3> --project`.
```
using Counters2, AutoTM, nGraph
# DenseNet
Counters2.experiment1lm(AutoTM.Experiments.large_densenet())

# Inception
Counters2.experiment1lm(AutoTM.Experiments.large_inception())

# Resnet
Counters2.experiment1lm(AutoTM.Experiments.large_resnet())
```
