#!/bin/sh
JL_PATH=~/projects/julia/julia

ARRAY_SIZE=29
#ARRAY_SIZE=32
MODE=2lm
ITERATIONS=10

JULIA_NUM_THREADS=24 numactl --physcpubind=24-47 --membind=1 $JL_PATH \
    --color=yes traffic.jl                      \
    --array-size $ARRAY_SIZE                    \
    --counter_type rw tags queues dram-queues   \
    --mode $MODE                                \
    --inner_iterations $ITERATIONS
