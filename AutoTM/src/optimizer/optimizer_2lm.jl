#####
##### 2LM "Optimizer"
#####

struct Optimizer2LM <: AbstractOptimizer{Rational{Int64}} end
Utils.getratio(::Optimizer2LM) = 0 // 1
name(::Optimizer2LM) = "2lm"

function factory(
        backend::nGraph.Backend{nGraph.CPU},
        func,
        opt::Optimizer2LM;
        # Useful for the GPU case
        adjust_io = false,
        defrag = true,
        just_profile = false,
        use_scratchpad = false,
        threshold = 20,
        profile_kw...
    )

    # Get the function, arguments, and keyword arguments from the provided function
    Profiler.enable_passes()
    f, args, kw = func()

    #A callback that profiles the ngraph function
    function cb(f::nGraph.NFunction)
        # Do some minor editing the order of nodes in the graph to hopefully yield slightly
        # better memory characteristics
        priority_pass!(f)

        if use_scratchpad
            data = Profiler.FunctionData(f, nGraph.CPU)
            setup_scratchpad!(data; threshold = threshold)
        end
    end

    callbacks = CallbackChain()
    callback!(callbacks, cb)

    fex = nGraph.compile(
        backend,
        f,
        args...;
        callback = callbacks,
        emit_timing = true,
        kw...
    )

    return fex
end

