#####
##### PMEM initialization
#####

"""
$(SIGNATURES)

Configure the PMM allocator in ngraph to allocated PMM files in `dir`.
Default: `/mnt/public`
"""
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

"""
$(SIGNATURES)

Set OMP thread affinities be setting appropriate environmental variables.

* `omp_num_threads`: The number of threads to reserve. Default: `24`
* `reserved_cores`: The number of physical CPU cores to reserve. Default: `omp_num_threads`
* `threads_per_core`: If `1`, essentially disables hyperthreading. If `2`, enables
    hyperthreading. Default: `1`.

The effects of this function are only valid if the ngraph JIT compiler has not yet been
invoked. Afterwards, no changes will take place.

This method is called during Module initialization, but may be called as many times as 
desired before the first ngraph compilation.

See also, [`teardown_affinities`](@ref)
"""
function setup_affinities(;
        omp_num_threads = 24,
        reserved_cores = omp_num_threads,
        threads_per_core = 1
    )

    if !in(threads_per_core, (1, 2))
        throw(ArgumentError("""
            Expected key word `threads_per_core` to be either 1 or 2.
            Instead, got $threads_per_core.
            """
        ))
    end

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
    ENV["OMP_MAX_ACTIVE_LEVELS"] = 2

    return nothing
end

_env_context() = (
    kmp_hw_subset = ENV["KMP_HW_SUBSET"]::String,
    omp_num_threads = ENV["OMP_NUM_THREADS"]::String,
)

"""
$(SIGNATURES)

Turn on ngraph's JIT feature.
"""
setup_codegen() = nGraph.enable_codegen()

"""
$(SIGNATURES)

Enable the ngraph compiler to reuse memory for intermediate tensors based on liveness 
analysis.
"""
setup_passes() = nGraph.set_pass_attributes(nGraph.ReuseMemory())
