#####
##### PMEM initialization
#####

function setup_pmem(dir = "/mnt/public/")
    if isdir(dir) 
        for file in readdir(dir)
            rm(joinpath(dir, file); recursive = true)
        end
    end

    manager = nGraph.Lib.getinstance()
    nGraph.Lib.set_pool_dir(manager, dir)
    return nothing
end

#####
##### Setup Affinities
#####

# For whatever reason, we have to set up the number of threads before the first 
# compilation of an nGraph executable.
#
# After that, changes in the number of threads won't be seen by nGraph
function setup_affinities(;omp_num_threads = 24, reserved_cores = omp_num_threads, threads_per_core = 1)
    if nGraph.have_compiled() == true
        @error """
        The nGraph compiler has already been invoked.

        The number of threads WILL NOT change.
        """
        return nothing
    end

    ENV["KMP_AFFINITY"] = "compact,granularity=fine"

    # Use the first 24 cores - 1 threads for each core
    # Send to numa-node 1 for a hopefully more quiet system
    #
    # See docs/src/runner/kmp.md for syntax documentation
    ENV["KMP_HW_SUBSET"] = "1s@1,$(reserved_cores)c,$(threads_per_core)t" 

    # 1 Threads for each core
    ENV["OMP_NUM_THREADS"] = omp_num_threads
    #ENV["OMP_NESTED"] = true
    ENV["OMP_MAX_ACTIVE_LEVELS"] = 2
    #ENV["OMP_THREAD_LIMIT"] = num_threads
    return nothing
end

teardown_affinities() = delete!.(Ref(ENV), ("KMP_AFFINITY", "KMP_HW_SUBSET", "OMP_NUM_THREADS"))

_env_context() = (
    kmp_hw_subset = ENV["KMP_HW_SUBSET"]::String,
    omp_num_threads = ENV["OMP_NUM_THREADS"]::String,
)

function setup_profiling()
    nGraph.enable_codegen()

    # Use the built in performance counter API instead of spitting out a JSON file
    #nGraph.enable_timing()
end

function setup_passes()
    nGraph.set_pass_attributes(nGraph.ReuseMemory())
end
