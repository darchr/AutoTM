# Include the `Benchmarker` code.
#
# Push through the `test_vgg` to make sure that whole pipeline works.
include("../../experiments/Benchmarker/src/Benchmarker.jl")
using .Benchmarker: Benchmarker

@testset "Testing Benchmarker" begin
    Benchmarker.kernel_profile(Benchmarker.test_vgg())
    Benchmarker.run_conventional(
        Benchmarker.test_vgg(),
        [AutoTM.Optimizer.Static, AutoTM.Optimizer.Synchronous, AutoTM.Optimizer.Numa],
        Benchmarker.common_ratios(),
    )

    # Generate Plots
    Benchmarker.plot_speedup(
        models = [Benchmarker.test_vgg()],
    )

    Benchmarker.plot_conventional_error(
        models = [Benchmarker.test_vgg()],
    )

    Benchmarker.plot_costs(
        pairs = [Benchmarker.test_vgg() => "synchronous"],
    )
end
