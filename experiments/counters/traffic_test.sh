#!/bin/sh

JULIA_NUM_THREADS=24 numactl --cpunodebind=1 --membind=1 \
    ~/julia-1.3.0-rc4/bin/julia --color=yes traffic.jl --array-size 1000000000 --counter_type rw tags queues dram-queues
