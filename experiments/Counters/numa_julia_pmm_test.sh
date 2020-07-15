#!/bin/sh
JL_PATH=~/projects/julia/julia

# JULIA_NUM_THREADS=4 numactl --physcpubind=24-47 --membind=1 $JL_PATH --project \
#     -e "using Counters; Counters.pmm_direct_test()"
#
# JULIA_NUM_THREADS=5 numactl --physcpubind=24-47 --membind=1 $JL_PATH --project \
#     -e "using Counters; Counters.pmm_direct_test()"
#
# JULIA_NUM_THREADS=8 numactl --physcpubind=24-47 --membind=1 $JL_PATH --project \
#     -e "using Counters; Counters.pmm_direct_test()"

# JULIA_NUM_THREADS=16 numactl --physcpubind=24-47 --membind=1 $JL_PATH --project \
#     -e "using Counters; Counters.pmm_direct_test()"

JULIA_NUM_THREADS=24 numactl --physcpubind=24-47 --membind=1 $JL_PATH --project \
    -e "using Counters; Counters.pmm_direct_test()"
