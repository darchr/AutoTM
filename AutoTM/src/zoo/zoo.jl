module Zoo

# Stdlib requirements
using Statistics, Random

# Use Flux for the actual modeling
using Flux
import nGraph

include("densenet.jl")
include("inception.jl")
include("resnet.jl")
include("transformer.jl")
include("vgg.jl")
include("dlrm.jl")

end
