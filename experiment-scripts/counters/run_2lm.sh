# Make sure `counters.jl` is running

## Without scratchpad option
# RW Counters
julia --color=yes runner.jl --mode=2lm --workload=large_vgg --counter_type=rw
julia --color=yes runner.jl --mode=2lm --workload=large_inception --counter_type=rw
julia --color=yes runner.jl --mode=2lm --workload=large_resnet --counter_type=rw
julia --color=yes runner.jl --mode=2lm --workload=large_densenet --counter_type=rw

# Tag Counters
julia --color=yes runner.jl --mode=2lm --workload=large_vgg --counter_type=tags
julia --color=yes runner.jl --mode=2lm --workload=large_inception --counter_type=tags 
julia --color=yes runner.jl --mode=2lm --workload=large_resnet --counter_type=tags 
julia --color=yes runner.jl --mode=2lm --workload=large_densenet --counter_type=tags 

# Queue counters
julia --color=yes runner.jl --mode=2lm --workload=large_vgg --counter_type=queue
julia --color=yes runner.jl --mode=2lm --workload=large_inception --counter_type=queue 
julia --color=yes runner.jl --mode=2lm --workload=large_resnet --counter_type=queue 
julia --color=yes runner.jl --mode=2lm --workload=large_densenet --counter_type=queue 


## With scratchpad option
# RW Counters
julia --color=yes runner.jl --mode=2lm --workload=large_vgg --counter_type=rw --use_2lm_scratchpad
julia --color=yes runner.jl --mode=2lm --workload=large_inception --counter_type=rw --use_2lm_scratchpad
julia --color=yes runner.jl --mode=2lm --workload=large_resnet --counter_type=rw --use_2lm_scratchpad
julia --color=yes runner.jl --mode=2lm --workload=large_densenet --counter_type=rw --use_2lm_scratchpad

# Tag Counters
julia --color=yes runner.jl --mode=2lm --workload=large_vgg --counter_type=tags --use_2lm_scratchpad
julia --color=yes runner.jl --mode=2lm --workload=large_inception --counter_type=tags --use_2lm_scratchpad
julia --color=yes runner.jl --mode=2lm --workload=large_resnet --counter_type=tags --use_2lm_scratchpad
julia --color=yes runner.jl --mode=2lm --workload=large_densenet --counter_type=tags --use_2lm_scratchpad

# Queue counters 
julia --color=yes runner.jl --mode=2lm --workload=large_vgg --counter_type=queue --use_2lm_scratchpad
julia --color=yes runner.jl --mode=2lm --workload=large_inception --counter_type=queue --use_2lm_scratchpad
julia --color=yes runner.jl --mode=2lm --workload=large_resnet --counter_type=queue --use_2lm_scratchpad
julia --color=yes runner.jl --mode=2lm --workload=large_densenet --counter_type=queue --use_2lm_scratchpad
