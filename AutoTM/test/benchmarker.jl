# Include the `Benchmarker` code.
#
# Push through the `test_vgg` to make sure that whole pipeline works.
include("../../experiments/Benchmarker/src/Benchmarker.jl")
using .Benchmarker: Benchmarker

@testset "Testing Benchmarker" begin

end
