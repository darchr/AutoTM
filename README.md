# AutoTM

| **Documentation** | **Archive** | **Status** |
|:---:|:---:|:---:|
[![][docs-latest-img]][docs-latest-url] | [![DOI](https://zenodo.org/badge/200740567.svg)](https://zenodo.org/badge/latestdoi/200740567) | [![][travis-img]][travis-url] |

Memory capacity is a key bottleneck for training large scale neural networks. 
Intel® OptaneTM DC PMM (persistent memory modules) which are available as NVDIMMs are a disruptive technology that promises significantly higher read bandwidth than traditional SSDs and significantly high capacity and at a lower cost per bit than traditional DRAM. 
In this work we will show how to take advantage of this new memory technology to reduce the overall system cost by minimizing the amount of DRAM required without compromising performance significantly. 
Specifically, we take advantage of the static nature of the underlying computational graphs in deep neural network applications to develop a profile guided optimization based on Integer Linear Programming (ILP) called AutoTM to optimally assign and move live tensors to either DRAM or NVDIMMs. 
Our approach can replace 80% of a system’s DRAM with PMM while only losing a geometric mean 27.7% performance. 
This is a significant improvement over first-touch NUMA, which loses 71.9% of performance. 
The proposed ILP based synchronous scheduling technique also provides 2x performance over 2LM (existing hardware approach where DRAM is used as a cache) for large batch size ResNet 200.

## Installation

For installation instructions, visit the documentation: http://arch.cs.ucdavis.edu/AutoTM/dev/installation/

## Building Documentation

To build documentation, run the following commands from the top directory
```sh
julia --project=docs/ -e 'using Pkg; Pkg.instantiate(); using Documenter; include("docs/make.jl")'
```
Documentation will be available at `docs/build/`.

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


[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: http://arch.cs.ucdavis.edu/AutoTM/dev/

[travis-img]: https://travis-ci.org/darchr/AutoTM.svg?branch=master
[travis-url]: https://travis-ci.org/darchr/AutoTM
