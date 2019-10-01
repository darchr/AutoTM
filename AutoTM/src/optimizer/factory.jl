# Default is to fallback to the inner call
#
# Do this fallback scheme to specialize for certain combinations of arguments if needed.
factory(args...; kw...) = _factory(args...; kw...)

# Ratio optimizers go through a refinement step
function factory(
        backend::nGraph.Backend{nGraph.CPU}, 
        func, 
        opt::AbstractOptimizer{Rational{Int64}}; 
        kw...
    )
    return ratiosearch(_factory, backend, func, opt; kw...)
end

struct CompilerRetry <: Exception end
function _factory(
        backend::nGraph.Backend,
        func,
        opt::T;
        # Useful for the GPU case
        adjust_io = false,
        defrag = true,
        just_profile = false,
        profile_kw...
    ) where {T <: AbstractOptimizer}

    # Get the function, arguments, and keyword arguments from the provided function
    Profiler.enable_passes()
    f, args, kw = func()

    # add a callback that will populate a reference to a `FunctionData` type
    frame_ref = Ref{Frame}()
    limits_ref = Ref{Vector{Int}}()
    creation_times = Float64[]
    optimization_times = Float64[]
    remote_args_ref = Ref{Set{TensorDescriptor}}()

    # A callback that profiles the ngraph function
    function cb(f::nGraph.NFunction)
        # Do some minor editing the order of nodes in the graph to hopefully yield slightly
        # better memory characteristics
        priority_pass!(f)
        data = profile(f, backend; profile_kw...)
        just_profile && throw(CompilerExit())

        # Initialize the node dram limits if needed
        if !isdefined(limits_ref, :x)
            # Get the limit from the optimizer
            # Find the input and output size of the function and subtract that from the
            # limit along with a fudge factor so we can fit the model on the GPU
            if adjust_io
                io_size = sum(sizeof, input_tensors(f)) + sum(sizeof, output_tensors(f))
                limit = max(getlimit(opt) - io_size, 0)
                opt_adjusted = _optimizer(opt, limit)
                modeltype = opt_adjusted(data, backend)
            else
                modeltype = opt(data, backend)
            end

            # Save the limits to the reference for reuse across defragmentation.
            limits_ref[] = modeltype.dram_limits
        else
            modeltype = opt(data, backend)
            modeltype.dram_limits = limits_ref[]
        end

        # Record statistics about how long solving takes.
        creation_time = @elapsed(frame = create_model(modeltype, data))
        optimization_time = @elapsed(optimize!(frame))

        # update the frame with the local args
        empty!(frame.local_args)
        for tensor in tensors(data) 
            islocalarg(frame, tensor) && push!(frame.local_args, tensor)
        end

        push!(creation_times, creation_time)
        push!(optimization_times, optimization_time)

        remote_args = configure!(f, frame)
        frame_ref[] = frame
        remote_args_ref[] = remote_args

        return nothing
    end

    # Defrag callback - if a function needs defragging, throws a `CompilerExit` exception to
    # avoid nGraph trying to allocate too much GPU memory
    function defrag_cb(f::nGraph.NFunction)
        if exceeds_limit(f, frame_ref[].modeltype, frame_ref[].local_args)
            # This is pretty ugly - sorry about that.
            modeltype = update(
                frame_ref[].modeltype, 
                frame_ref[].local_args, 
                profile(f, backend; profile_kw...)
            )
            limits_ref[] = modeltype.dram_limits

            throw(CompilerRetry())
        end
    end

    # Create a function to let the nGraph.jl compiler know if a function parameter or 
    # output is supposed to be remote
    isremote(x::nGraph.Node) = in(first(outputs(x)), remote_args_ref[])

    # Compile the function to a ngraph executable
    local fex
    retry = true
    while retry
        retry = false

        # Setup callbacks
        #
        # If the function needs defragging, a `CompilerExit` exception will be thrown and we
        # will have to try again.
        callbacks = CallbackChain()
        callback!(callbacks, cb)
        defrag && callback!(callbacks, defrag_cb)

        try
            #pre_args = nGraph._compile_snoop(backend, f, args...; isremote = isremote)

            fex = nGraph.compile(
                backend,
                f,
                args...;
                callback = callbacks,
                emit_timing = true,
                #isremote = isremote,
                kw...
            )
        catch e
            isa(e, CompilerRetry) || rethrow(e)
            retry = true
        end
    end

    # Following compilation, set up the appropriate arguments to live in persistent memory 
    # or not.
    remote_args = remote_args_ref[]

    metadata = Dict(
        :creation_times => creation_times,
        :optimization_times => optimization_times,
    )

    return fex, frame_ref[], metadata
end

#####
##### Ratio Searching
#####

"""
    ratiosearch(f, backend, func, opt; search_ratio = true, refinements = 7, kw...)

`f` - 
`func` - An AutoTM compatible function constructor

Further keywords get passed to the inner call to `f`.
"""
function ratiosearch(f, backend, func, opt; search_ratio = true, refinements = 7, kw...)
    @info "Trying Ratio $(getratio(opt))"

    # Just return the inner factory if we aren't interesting in performing a binary search
    # for the actual ratio to input that will return the desired ratio
    search_ratio || return f(backend, func, opt; kw...)

    # Perform a binary search
    ret = f(backend, func, opt; kw...)
    fex = first(ret)
    args = Base.tail(ret)

    # If we're within the desired tolerance, just return like normal
    checkmargin(fex, opt) && return (fex, args...)

    # For now, I'm assuming that the ratio generated by the optimized graph will always
    # be greater than the desired ratio due to defragmentation.
    mul = getratio(fex) > getratio(opt) ? 1 : -1

    # Start doing a grid search
    #
    # The function desired_ratio -> actual_ratio is not necessarily monotonic, so we can't
    # use a binary search. Instead this grid search thing seems to work alright.
    best_ratio = getratio(opt)
    best_err = geterr(fex, opt)
    current_ratio = best_ratio

    best_fex = fex
    best_args = args

    for i in 1:refinements
        # Use a step size starting with 1 and increasing or decreasing by the step size
        # until the ratio crosses the boundary of what we want.
        step = 1 // (2 ^ (i))
        @info """
        ------------------------
        Performing Refinement Iteration $i
        Step: $step
        ------------------------
        """

        for _ in 1:1
            current_ratio = current_ratio - (mul * step)
            current_ratio < 0 && break

            @info "Trying Ratio: $(convert(Float64, current_ratio))"
            ret = f(backend, func, _optimizer(opt, current_ratio); kw...)
            fex = first(ret)
            args = Base.tail(ret)

            @show typeof(fex)
            @show typeof(args)

            # If the ratios switch sign, time to exit
            if mul == 1
                getratio(fex) <= getratio(opt) && break
            elseif mul == -1
                getratio(fex) >= getratio(opt) && break
            end

            current_err = geterr(fex, opt)
            @info """
            Current Ratio: $(convert(Float64, current_ratio))
            Best Ratio: $(convert(Float64, best_ratio))

            Current Error: $(convert(Float64, current_err))
            Best Error: $(convert(Float64, best_err))
            """
            if current_err < best_err
                best_ratio = current_ratio
                best_err = current_err
                best_fex = fex
                best_args = args
            end
        end
        current_ratio = best_ratio
    end

    return (best_fex, best_args...)
end
