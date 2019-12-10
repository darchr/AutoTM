module Zoo

# Stdlib requirements
using Statistics, Random

using ..Utils: Actualizer, ForwardLoss

# Use Flux for the actual modeling
using Flux
import nGraph

# Like Flux's Chain, but not type stable.
struct UnstableChain
    x::Vector{Any}
end
Flux.@functor UnstableChain

function (U::UnstableChain)(x)
    for f in U.x
        x = f(x)
    end
    return x
end


include("densenet.jl")
include("inception.jl")
include("resnet.jl")
include("transformer.jl")
include("vgg.jl")
include("dlrm.jl")

end
