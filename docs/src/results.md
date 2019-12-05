# Results Data Structure

For convenience of generating plots, result data from benchmarking runs is stores as a serialized Julia data structure.
The details of that data structure are provided here.
Results themselves can be found in either `$AUTOTM_HOME/experiments/Benchmarker/data/cpu` or `$AUTOTM_HOME/experiments/Benchmarker/data/gpu`.

The top level struct is a Julia NamedTuple with the following fields
* `io_size`: The size in bytes of a network's input and output tensors.
* `default_alloc_size`: The number of bytes required by ngraph to run the graph natively.
* `gpu_managed_runtime`: The runtime of a network when using cudaMallocManaged (only applies to networks run on the GPU)
* `runs`: A vector containing result data for all benchmark runs for this particular workload.
    The item that varies between runs is the amount of DRAM allowed.
    The elements of this vector are of type `Dict{Symbol, Any}`.

## CPU

For CPU workloads, the metrics recorded in the `runs` dictionaries are

* `:creation_times`: Time spent creating the ILP formulation.
    This is a vector which may have multiple elements if the ILP was run multiple times due to defragmentation.

* `:optimization_times`: Time spent solving the ILP.
    Like `creation_times`, this may have multiple entries.

* `:predicted_runtime`: Runtime predicted by the ILP

* `:dram_limit`: The DRAM limit passed to the optimizer.

* `:tensor_size_map`: A dictionary mapping intermediate tensor names to their size in bytes.

* `:config_map`: A dictionary mapping ngraph nodes to their input and output configuration.

* `:ratio`: The ratio of PMM to DRAM.

* `:num_move_nodes`: The number of move nodes emitted.

* `:num_pmem_move_nodes`: The number of move nodes moving data from DRAM to PMM.
* `:num_dram_move_nodes`: The nubmer of move nodes moving data from PMM to DRAM.

* `:bytes_moved`: The total amount of data in bytes moved between memory pools.
* `:bytes_moved_pmem`: The number of bytes moved from DRAM to PMM.
* `:bytes_moved_dram`: The number of bytes moved from PMM to DRAM.

* `:num_async_move_nodes`: The number of asynchronous move nodes generated.
* `:num_pmem_async_move_nodes`: The number of asynchronous move nodes from DRAM to PMM.
* `:num_dram_async_move_nodes`: The number of asynchronous move nodes from PMM to DRAM.

* `:bytes_async_moved`: The total amount of data in bytes moved asynchronously.
* `:bytes_async_moved_pmem`: The amount of data in bytes moved asynchronously from DRAM to PMM.
* `:bytes_async_moved_dram`: The amount of data in bytes moved asynchronously from PMM to DRAM.

* `:num_kernels`: The number of ngraph nodes in the computation graph.
* `:num_input_tensors`: The total number of kernel inputs in the computation graph.
* `:num_output_tensors`: The total number of kernel outputs in the computation graph.

* `:num_dram_input_tensors`: The number of kernel inputs that are in DRAM.
* `:num_dram_output_tensors`: The number of kernel outputs that are in DRAM.

* `:bytes_input_tensors`: The total size of all kernel inputs.
* `:bytes_output_tensors`: The total size of all kernel outputs.

* `:bytes_dram_input_tensors`: The total size of all kernel inputs that are in DRAM.
* `:bytes_dram_output_tensors`: The total size of all kernel outputs that are in DRAM.

* `:dram_alloc_size`: The actual allocation size made by ngraph for DRAM.
* `:pmem_alloc_size`: The actual size of the PMM pool allocated by ngraph.
* `:move_time`: Estimate of time spent moving data.
    Estimated based on the number of move nodes and the expected time for each move node.

If the workload was run, the following fields will also be included

* `:actual_runtime`: The actual measured runtime of the workload.
* `:kernel_times`: A dictionary mapping kernel names to their actual runtime.

## GPU

The entries in the GPU dictionary are largely the same. 
In the case of the GPU, the term `pmem` refers to host DRAM and `dram` refers to device DRAM.
Additionally, the GPU data has the following entry:

* `:oracle_time`: Predicted fastest runtime if all kernels with selectable implementations used their fastest implementation.
