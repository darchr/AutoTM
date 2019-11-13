#!/bin/sh
JL_PATH=~/projects/julia/julia
JULIA_NUM_THREADS=24 numactl --physcpubind=24-47 --membind=1 $JL_PATH --project
