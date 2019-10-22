#!/bin/sh

# Make sure `counters.jl` is running
julia --color=yes runner.jl --mode=1lm \
    --workload large_vgg large_inception large_resnet large_densenet \
    --counter_type rw queues insert-check

