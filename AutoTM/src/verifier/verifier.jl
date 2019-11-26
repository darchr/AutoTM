module Verifier

using Random
using ..Utils
using ..Optimizer

using ProgressMeter: @showprogress
using nGraph

export verify

# We need to be able to verify that the output of modified graphs matches that of a standard
# graph.
#
# To do that, we use an auxiliary function that accepts the constructor function for a model
# as well as any passes to be performed on the model and checks that the base computation
# matches the modifed computation.

# Get all of the input and output parameters of an executable.
io_arrays(fex) = vflatten(nGraph.splat_inputs(fex), nGraph.splat_outputs(fex))

function verify(backend::nGraph.Backend{nGraph.CPU}, f, opt, cache;
        seed = 8086,
        inner_iterations = 5,
        outer_iterations = 2,
    )

    success = true
    for i in 1:outer_iterations
        println("Working on iteration $i")

        # Create a new seed for each outer iteration.
        outer_seed = rand(UInt64) 

        # Wrapped the callable in a seed generator
        seeded_f = @closure begin
            Random.seed!(outer_seed)
            return f()
        end

        # Create a baseline copy and an optimized copy of the function.
        fex_baseline = actualize(backend, seeded_f)
        fex_optimized = first(Optimizer.factory(
            backend, 
            seeded_f, 
            opt; 
            cache = cache, 
            search_ratio = false
        ))

        # Sanity check - make sure all input tensors are equivalent.
        # Because some output tensors may be undefined - don't check those until the
        # executables are run for the first time.
        for (a, b) in zip(nGraph.splat_inputs(fex_baseline), nGraph.splat_inputs(fex_optimized))
            if parent(a) != parent(b)
                equal_error(i, 0, seed, outer_seed)
                return false
            end
        end
        
        for j in 1:inner_iterations
            println("    Inner Iteration $j")

            # Run both functions - make sure ALL input and output tensors match each itertion.
            @time fex_baseline() 
            @time fex_optimized()
            for (a,b) in zip(io_arrays(fex_baseline), io_arrays(fex_optimized))
                # Check for NaN's
                if any(isnan, parent(a)) || any(isnan, parent(b))
                    nan_error(i, j, seed, outer_seed)
                    return false
                end

                # Check for Subnormal
                if any(issubnormal, parent(a)) || any(issubnormal, parent(b))
                    subnormal_error(i, j, seed, outer_seed)
                    return false
                end

                # Check equality
                #
                # Use `isapprox` because of non-determinism in GPU computations.
                if !isapprox(parent(a), parent(b))
                    equal_error(i, j, seed, outer_seed)
                    return (a, b)
                end
            end
        end
    end
    return true
end

# For the GPU, nondeterminism means we have to trace over a period of time.
function verify(backend::nGraph.Backend{nGraph.GPU}, f, opt, cache;
        seed = 8086,
        inner_iterations = 100,
        outer_iterations = 2,
    )

    success = true
    for i in 1:outer_iterations
        println("Working on iteration $i")

        # Create a new seed for each outer iteration.
        outer_seed = rand(UInt64) 

        # Wrapped the callable in a seed generator
        seeded_f = @closure begin
            Random.seed!(outer_seed)
            return f()
        end

        # GC fence everything to ensure we don't run out of memory.
        GC.gc()
        baseline = run_baseline(backend, seeded_f, opt, cache, inner_iterations)
        GC.gc()
        optimized = run_optimized(backend, seeded_f, opt, cache, inner_iterations)
        GC.gc()
        return baseline, optimized
    end
end

# Inner functions for GC purposes.
function run_baseline(backend, f, opt, cache, inner_iterations)
    fex = actualize(backend, f)
    vals = []
    @showprogress 1 for j in 1:inner_iterations
        push!(vals, collect(parent(fex())))
    end
    return vals
end

function run_optimized(backend, f, opt, cache, inner_iterations)
    #fex = actualize(backend, f)
    fex = first(Optimizer.factory(
        backend, 
        f, 
        opt; 
        cache = cache, 
    ))

    vals = []
    @showprogress 1 for j in 1:inner_iterations
        push!(vals, collect(parent(fex())))
    end
    return vals
end

#####
##### Error Generation.
#####

nan_error(x...) = generic_error("NaN Values", x...)
subnormal_error(x...) = generic_error("Subnormal Values", x...)
equal_error(x...) = generic_error("Match Error", x...)

function generic_error(prefix, outer_iteration, inner_iteration, seed, outer_seed)
    message = """
    $prefix
    Outer Iteration: $outer_iteration
    Inner Iteration: $inner_iteration
    Master seed: $seed
    Seed for this round: $outer_seed
    """
    printstyled(message; color = :red)
    println()
    return nothing
end

end
