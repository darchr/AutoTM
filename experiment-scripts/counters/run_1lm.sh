# Make sure `counters.jl` is running
julia --color=yes runner.jl --mode=1lm --workload=large_vgg --counter_type=rw
julia --color=yes runner.jl --mode=1lm --workload=large_inception --counter_type=rw
julia --color=yes runner.jl --mode=1lm --workload=large_resnet --counter_type=rw
julia --color=yes runner.jl --mode=1lm --workload=large_densenet --counter_type=rw

julia --color=yes runner.jl --mode=1lm --workload=large_vgg --counter_type=queue
julia --color=yes runner.jl --mode=1lm --workload=large_inception --counter_type=queue
julia --color=yes runner.jl --mode=1lm --workload=large_resnet --counter_type=queue
julia --color=yes runner.jl --mode=1lm --workload=large_densenet --counter_type=queue
