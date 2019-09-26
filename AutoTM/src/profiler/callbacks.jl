# Callbacks for merging tensor groups and generating "inplace" annotations.

# Partial application.
fex_cb!(fex::nGraph.FluxExecutable) = x -> fex_cb!(x, fex)

# Implementations
#
# For general FluxExecutables - we probably don't want to run any kind of merging pass.
# But when the SGD optimizer, we want to associate the double buffering inputs and outputs
fex_cb!(data, fex::nGraph.FluxExecutable{T, nGraph.InferenceState}) where {T} = nothing
function fex_cb!(data, fex::nGraph.FluxExecutable{T, nGraph.SGDState}) where {T}
    # Create a dictionary mapping TensorDescriptors to their relevant XTensors
    descriptor_to_xtensor = Dict(unx(t) => t for t in tensors(data)) 

    # Run through the inputs and outputs of the optimizer - merging them into the same 
    # group.
    #
    # If the underlying pointer is the same, also mark them as `inplace`.
    optimizer = fex.optimizer
    iter = (
        optimizer.inputs,
        optimizer.outputs,
        optimizer.input_descriptors,
        optimizer.output_descriptors,
    )
    for (i, o, id, od) in zip(iter...)
        xi = descriptor_to_xtensor[id]
        xo = descriptor_to_xtensor[od]
        merge!(xi, xo)
        if nGraph.rawptr(i) == nGraph.rawptr(o)
            @info "Making in inplace annotation: $(nGraph.name(id)) -- $(nGraph.name(od))"
            makeinplace(xi)
            makeinplace(xo)
        end
    end
    return nothing
end
