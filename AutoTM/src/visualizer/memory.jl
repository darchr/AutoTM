# Tool for inspecting and generating the heap allocation of an executable.
#
# The modified nGraph allows us to pass two Julia callbacks to the compilation process.
# The first callback will be called before the memory allocation pass, and the second will
# be called AFTER memory allocatio pass.
#
# Since our graphs may exceed the physical memory of a system - we will just insert the
# extraction of the intermediate tensors from the second callback.
struct TensorRecord
    start_index::Int
    stop_index::Int
    offset::Int
    height::Int
end

function Base.isless(a::TensorRecord, b::TensorRecord)
    # First - order by starting index.
    if a.start_index < b.start_index
        return true

    # Then, order by offset
    elseif a.start_index == b.start_index
        if a.offset < b.offset
            return true
        end
    end
    return false
end

# PGF Draw Rectangle
_draw_rect(x0, y0, x1, y1) = "\\draw [fill = black,fill = black,  black, thick] (axis cs:$x0,$y0) rectangle (axis cs:$x1,$y1);"
_draw_rect(r::TensorRecord) = _draw_rect(r.start_index, r.offset, r.stop_index, r.offset + r.height)

# Unfortunately, we have to carry around this cache ...
function heap_plot(func, cache::Profiler.AbstractKernelCache; use_scratchpad = false)
    # Create a record for each tensor in the compiled graph.
    tensor_records = _get_records(func, cache; use_scratchpad = use_scratchpad)
    sort!(tensor_records)

    # Plot a rectangle for each tensor.
    rectangle_strings = _draw_rect.(tensor_records)

    # Get the axis min and maxes
    ymin = 0
    ymax = maximum(x -> x.offset + x.height, tensor_records)
    xmin = 0
    xmax = maximum(x -> x.stop_index, tensor_records)

    axs = @pgf Axis(
        {
            width = "20cm",
            height = "14cm",
            ymin = ymin,
            ymax = ymax,
            xmin = xmin,
            xmax = xmax,
        },
        rectangle_strings...,
    )
    return axs
end

# Create a tensor record for each tensor in the compiled nGraph function.
function _get_records(func, cache; use_scratchpad = false)
    # For now - always use the CPU backend.
    backend = nGraph.Backend("CPU")

    # Instantiate the function callable
    f, args, kw = func()

    # Record of tensors extracted from the nGraph function.
    tensor_records = TensorRecord[]

    # The first callback that we provide is just empty.
    if use_scratchpad
        first_cb = function(f::nGraph.NFunction)
            Optimizer.priority_pass!(f)
            data = Profiler.profile(f, backend; cache = cache)
            Optimizer.setup_scratchpad!(data; threshold = 100)
        end
    else
        first_cb = (f::nGraph.NFunction) -> Optimizer.priority_pass!(f)
    end

    function cb(f::nGraph.NFunction)
        # Perform a profiling pass. We don't actually NEED to timing profile information,
        # but the `FunctionData` struct and its live tensor iterator will be very helpful
        # for extracting the data we want.
        data = Profiler.profile(f, backend; cache = cache)

        for tensor in Profiler.tensors(data)
            # Extract the stop and start index for this tensor.
            start_index = Profiler.producer(tensor).index
            stop_index = Profiler.consumer(tensor).index

            # Get the height (allocation bytes) of the tensor as well as its assigned offset.
            height = sizeof(tensor)
            offset = Int(Profiler.getoffset(tensor))

            record = TensorRecord(
                start_index,
                stop_index,
                offset,
                height,
            )
            push!(tensor_records, record)
        end

        # Exit from compilation
        throw(Profiler.CompilerExit())
    end

    # Setup the callback chains and begin the compilation.
    callbacks = CallbackChain()
    callback!(callbacks, first_cb)
    callback!(callbacks, cb)

    try
        fex = nGraph.compile(
            backend,
            f,
            args...;
            callback = callbacks,
            kw...,
        )
    catch e
        isa(e, CompilerExit) || rethrow(e)
    end

    return tensor_records
end
