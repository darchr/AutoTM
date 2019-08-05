#!/bin/bash

# Run 8, 16, then 24 threads
#numactl --cpunodebind=1 julia kernel_benchmark.jl --nthreads 8 --datafile ../kernel_data.jls --refresh
#numactl --cpunodebind=1 julia kernel_benchmark.jl --nthreads 16 --datafile ../kernel_data.jls
numactl --cpunodebind=1 julia kernel_benchmark.jl --nthreads 24 --datafile ../kernel_data.jls --refresh
