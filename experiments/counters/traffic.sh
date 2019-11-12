#!/bin/sh
JL_PATH=~/projects/julia/julia

ARRAY_SIZE=29
#ARRAY_SIZE=32
#ARRAY_SIZE=34
MODE=2lm
ITERATIONS=10

# Core counters
JULIA_NUM_THREADS=24 numactl --physcpubind=24-47 --membind=1 $JL_PATH \
    --color=yes traffic.jl                      \
    --array-size $ARRAY_SIZE                    \
    --counter_region core                       \
    --mode $MODE                                \
    --inner_iterations $ITERATIONS

# Uncore counters
JULIA_NUM_THREADS=24 numactl --physcpubind=24-47 --membind=1 $JL_PATH \
    --color=yes traffic.jl                      \
    --array-size $ARRAY_SIZE                    \
    --counter_region uncore                     \
    --counter_type rw tags                      \
    --mode $MODE                                \
    --inner_iterations $ITERATIONS

