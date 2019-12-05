# Experiment Customization

There are a couple of ways to customize experiments.

## CPU - Changing the Number of Threads.

The CPU portion of the code defaults to using 24 threads on Socket 1 of a dual socket system (with sockets numbered as 0 and 1).
This can be changed by calling
```
AutoTM.setup_affinities(; omp_num_threads = nthreads)
```
By default, thread affinities are assigned one per physical core, essentially disabling hyperthreading.
If you really want hyperthreading, the keyword argument `threads_per_core = 2` may be passed to `setup_affinities`.

!!! note

    Calls to `setup_affinities` only work if called before the first run of the ngraph compiler.
    The LLVM compiler backing ngraph doesn't like changing the number of OMP threads for some reason.

    Also don't do something crazy like `omp_num_threads = 1024` - I have no idea what will happen.

Kernel profiles are parameterized by number of threads so you don't have to worry about kernel profiles clobbering when changing the number of threads.

## CPU - Changing DRAM Limits

Supporting different PMM to DRAM ratios is straight forward.
When calling [`run_conventional`](@ref PMM - Conventional Benchmarks) entry function, custom ratios may be passed.
These ratios are simply defined by Julia's native `Rational{Int}` type.
For example, for a 16 to 1 PMM to DRAM ratio, simply pass `16 // 1`.
The resulting call might look like
```julia
Benchmarker.run_conventional(
    Benchmarker.test_vgg(),
    [AutoTM.Optimizer.Synchronous],
    16 // 1
)
```

Furthermore, a hard DRAM limit can be passed by just passing in an `Int` number of bytes to the third argument.

## GPU - Changing DRAM Limits

The GPU DRAM limits can be changed by changing the `GPU_MAX_MEMORY` and `GPU_MEMORY_OVERHEAD` variables as described in [GPU](@ref).

## CPU/GPU - New Networks

The code to create the benchmarked networks lives in `$AUTOTM_HOME/AutoTM/src/zoo`.
Networks are modeled following Julia's [Flux](https://github.com/FluxML/Flux.jl) machine learning library and are converted into ngraph computation graphs (with the help of [nGraph.jl](https://github.com/hildebrandmw/nGraph.jl) and the mighty [Cassette](https://github.com/jrevels/Cassette.jl)).

Custom networks can be defined externally and passed as an `AutoTM.Actualizer` to the functions `Benchmarker.run_conventional` or `Benchmarker.run_gpu`. A detailed example is given below.

Suppose we want to model a simple MLP.
```julia
using Benchmarker, Flux, AutoTM, nGraph

# Define a function that returns a simple MLP wrapped up in an `Actualizer`.
function mlp(batchsize)
    # Define the network
    network = Flux.Chain(
        Dense(4096, 4096, Flux.relu),
        Dense(4096, 4096, Flux.relu),
        Dense(4096, 4096, Flux.relu),
        Dense(4096, 10, Flux.relu),
        softmax,
    ) 

    # Create input array
    X = randn(Float32, 4096, batchsize)

    # Create dummy one-hot input
    Y = zeros(Float32, 10, batchsize)
    for i in 1:batchsize
        Y[rand(1:10)] = one(eltype(Y))
    end

    # Compute the loss function.  
    loss(x, y) = Flux.crossentropy(network(x), y)
    return AutoTM.Actualizer(loss, X, Y; optimizer = nGraph.SGD(Float32(0.005)))
end

# This function can now be passed to Benchmarker.run_gpu
# If running with a batchsize of 16
Benchmarker.run_gpu(() -> mlp(16))
```
The results from the above will end up in `$AUTOTM_HOME/experiments/Benchmarker/data/gpu` with the name `unknown_network`.
Results can be expected by deserializing the data

```julia
using Serialization
data = deserialize("data/gpu/unknown_network_asynchronous_gpu_profile.jls");
display(first(data.runs))

# Roughly Expected Output
#   :bytes_async_moved_dram    => 2003336
#   :bytes_input_tensors       => 545962020
#   :predicted_runtime         => 0.41705
#   :pmem_alloc_size           => 0x000000000012b000
#   :num_async_move_nodes      => 32
#   :num_dram_async_move_nodes => 17
#   :move_time                 => 0.0
#   :dram_alloc_size           => 269910080
#   :num_input_tensors         => 107
#   :num_dram_move_nodes       => 17
#   :actual_runtime            => 0.00424737
#   :bytes_output_tensors      => 744613584
#   :bytes_async_moved_pmem    => 2002056
#   :num_dram_input_tensors    => 107
#   :tensor_size_map           => Dict(...)
#   :num_dram_output_tensors   => 72
#   :bytes_moved_pmem          => 2002056
#   :num_pmem_async_move_nodes => 15
#   :num_kernels               => 72
#   :bytes_async_moved         => 4005392
#   :bytes_moved               => 0
#   :dram_limit                => 8597
#   :bytes_dram_input_tensors  => 545962020
#   :bytes_dram_output_tensors => 744613584
#   :bytes_moved_dram          => 2003336
#   :num_move_nodes            => 0
#   :num_output_tensors        => 72
#   :num_pmem_move_nodes       => 15
#   :oracle_time               => 4140.0
```

