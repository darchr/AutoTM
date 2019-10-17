# TODO: Revive
module Verifier

export verify

# We need to be able to verify that the output of modified graphs matches that of a standard
# graph.
#
# To do that, we use an auxiliary function that accepts the constructor function for a model
# as well as any passes to be performed on the model and checks that the base computation
# matches the modifed computation.
#
# NOTES:
# - Need to set the random seed before each call to the model constructor to ensure that
#   the model enters a known and consistent state before any modification.
#
# - Need to provide utility functions to extract the before/after parameters of each
#   function for comparison.

astuple(x::Tuple) = x
astuple(x) = (x,)

"""
- `f`: Function `() -> fex, args` that constructs a model and its arguments.

- `pass`: Function `(FluxExecutable) -> FluxExecutable`: Takes an executable and returns
    a modified executable that will be compared against the baseline executable.

- `env`: Tuple of environmental "var"=>val to forward to the first call to `actualize`.
    Main use is turning on CudaMallocManaged for comparison purposes.
"""
function verify(backend, f, opt; seed = 8086, env = (), iterations = 1, rtol = 0.05)
    # Wrap the function in another function that sets the random seed - ensuring the same
    # parameter values every time.
    f_wrapped = (args...; kw...) -> (Random.seed!(seed); return f(args...; kw...))

    # Get the reference inputs/outputs as well as the inputs from the optimized version.
    #
    # We expect all to be approximately the same.
    # Call the GC inbetween to free resources. This is important when running on the GPU
    # because otherwise, we'll get an `out of memory` error.
    GC.gc()
    ref_inputs, ref_outputs = _baseline(backend, f_wrapped, seed, env, iterations)
    GC.gc()
    #opt_inputs, opt_outputs = _baseline(backend, f_wrapped, seed, env, iterations)
    opt_inputs, opt_outputs = _test(backend, f_wrapped, opt, seed, iterations)
    GC.gc()

    # Perform all of the checks on the inputs and outputs
    passed = true 

    # Check that the reference inputs and outputs are not NAN or Subnormal
    for (i, (inputs, outputs)) in enumerate(zip(ref_inputs, ref_outputs))
        if any(x -> any(isnan, x), inputs)
            @error "Reference Inputs at Iteration $i are NaN!"
            passed = false
        end
        if any(x -> any(issubnormal, x), inputs)
            @error "Reference Inputs at Iteration $i are Subnormal!"
            passed = false
        end

        if any(x -> any(isnan, x), outputs)
            @error "Reference Inputs at Iteration $i are NaN!"
            passed = false
        end
        if any(x -> any(issubnormal, x), outputs)
            @error "Reference Inputs at Iteration $i are Subnormal!"
            passed = false
        end
    end

    # Check the inputs and outputs of the optimized results for NaN or Subnormal
    for (i, (inputs, outputs)) in enumerate(zip(opt_inputs, opt_outputs))
        if any(x -> any(isnan, x), inputs)
            @error "Reference Inputs at Iteration $i are NaN!"
            passed = false
        end
        if any(x -> any(issubnormal, x), inputs)
            @error "Reference Inputs at Iteration $i are Subnormal!"
            passed = false
        end

        if any(x -> any(isnan, x), outputs)
            @error "Reference Inputs at Iteration $i are NaN!"
            passed = false
        end
        if any(x -> any(issubnormal, x), outputs)
            @error "Reference Inputs at Iteration $i are Subnormal!"
            passed = false
        end
    end

    # Check reference and optimized for approximate equality
    iter = zip(ref_inputs, ref_outputs, opt_inputs, opt_outputs)
    for (i, (ref_input, ref_output, opt_input, opt_output)) in enumerate(iter)
        # Check outputs
        for (r, o) in zip(ref_output, opt_output)
            if !isapprox(r, o; rtol = rtol)
                @error "Reference and Optimized Output not equal on iteration $i"
                passed = false
            end
        end

        # Check inputs
        for (r, o) in zip(ref_input, opt_input)
            if !isapprox(r, o; rtol = rtol)
                @error "Reference and Optimized Input not equal on iteration $i"
                printstyled("Reference Input"; color = :red)
                display(r)
                printstyled("Optimized Input"; color = :red)
                display(o)
                passed = false
            end
        end
    end

    if !passed
        @error "Test did not pass for optimizer: $(typeof(opt))" opt
    end

    # Just show the losses
    for (r,o) in zip(ref_outputs, opt_outputs)
        printstyled("Reference Output"; color = :red)
        display(first(r))
        printstyled("Optimized Output"; color = :red)
        display(first(o))
    end

    return passed
end

# Run the reference version of the network
function _baseline(backend, f, seed, env, iterations)
    fex = actualize(backend, f; env = env)

    # Keep track on the CPU of the inputs and outputs across all iterations
    inputs = []
    outputs = []

    for _ in 1:iterations
        fex()
        push!(inputs, read.(nGraph._splat_inputs(fex)))
        push!(outputs, read.(nGraph._splat_outputs(fex)))
    end
    return inputs, outputs
end

function _test(backend, f, opt, seed, iterations)
    fex = first(factory(backend, f, opt))

    inputs = []
    outputs = []

    for _ in 1:iterations
        fex()
        push!(inputs, read.(nGraph._splat_inputs(fex)))
        push!(outputs, read.(nGraph._splat_outputs(fex)))
    end
    return inputs, outputs
end

#####
##### Track Training
#####

function track(backend, f, opts; 
        seed = 8086, 
        env = (), 
        inner_iterations = 300, 
        outer_iterations = 1,
    )

    # Wrapped the callable in a seed generator
    f_wrapped = () -> (Random.seed!(seed); return f()) 

    results = Dict{Any, Any}()

    # Run the baseline
    for _ in 1:outer_iterations
        f_wrapped2 = () -> actualize(backend, f; env = env)
        GC.gc()
        losses = _track(backend, f_wrapped2, inner_iterations)
        
        arr = get!(results, "baseline", [])
        push!(arr, losses)
    end

    # Run each of the optimizations
    for opt in opts 
        for _ in 1:outer_iterations
            f_wrapped2 = () -> first(factory(backend, f, opt))
            GC.gc() 
            losses = _track(backend, f_wrapped2, inner_iterations)
            
            arr = get!(results, opt, [])
            push!(arr, losses)
        end
    end

    return results
end

function _track(backend, f, iterations)
    fex = f()
    results = Vector{Float32}(undef, iterations)
    @showprogress 1 for i in 1:iterations
        results[i] = read(fex())[]
    end
    return results
end

end
