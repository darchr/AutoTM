#!/bin/sh

# Please make sure `counters.jl` is running

## Without scratchpad option
# RW Counters
julia --color=yes runner.jl --mode=2lm --workload large_vgg large_inception large_resnet large_densenet --counter_type rw queues insert-check tags

## With scratchpad option
julia --color=yes runner.jl --mode=2lm --workload large_vgg large_inception large resmet large_densenet --counter_type rw queues insert-check tags --use_2lm_scratchpad
