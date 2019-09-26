# Things are starting to get complicated enough that I feel like I need some tests for
# this ...
import AutoTM.Profiler: nodes, unx
import AutoTM.Utils: IOConfig, DRAM, PMEM, inputs, outputs

function merge_cb(data)
    node = nodes(data)[findfirst(x -> nGraph.description(unx(x)) == "Add", nodes(data))]
    io = vcat(inputs(node), outputs(node))
    merge!(io)

    # Test that all elements in "io" belong to the same group.
    @test isone(length(unique(map(x -> x.group, io))))
    return nothing
end

@testset "Testing Paired Parameters" begin
    backend = nGraph.Backend("CPU")
    cache = AutoTM.Profiler.CPUKernelCache(joinpath(@__DIR__, "cache.bin"); force_new = true)
    # First - test two of the same inputs to a node
    x = nGraph.parameter(Float32, (10, 10))
    f = x -> x + x
    fex = nGraph.compile(backend, f, x)

    # The code should generate such that the input is used twice for the sum.
    # We need to:
    # 1. Find the "Add" node in the compiled graph.
    # 2. Get a list of configs for it.
    # 3. Verify that the inputs to the add node don't vary since they come from the same
    #    input tensor
    data = AutoTM.Profiler.profile(fex; recache = true, cache = cache)
    node = nodes(data)[findfirst(x -> nGraph.description(unx(x)) == "Add", nodes(data))]

    # Now - we check the possible configs for the `Add` node
    configs = sort(AutoTM.Profiler.possible_configs(node))
    expected = [
        IOConfig((DRAM, DRAM), (DRAM,)),
        IOConfig((DRAM, DRAM), (PMEM,)),
        IOConfig((PMEM, PMEM), (DRAM,)),
        IOConfig((PMEM, PMEM), (PMEM,)),
    ]
    @test configs == expected

    # Next - we'll do the same things - but this time we'll mark the output of the addition
    # as belonging to the same group as the input - this should result in all inputs or
    # outputs as being in PMEM or DRAM.
    fex = nGraph.compile(backend, +, x, x)

    # Add a profiling callback to do the merging
    data = AutoTM.Profiler.profile(fex; recache = true, cache = cache, cb = merge_cb)
    node = nodes(data)[findfirst(x -> nGraph.description(unx(x)) == "Add", nodes(data))]

    # Now - we check the possible configs for the `Add` node
    configs = sort(AutoTM.Profiler.possible_configs(node))
    expected = [
        IOConfig((DRAM, DRAM), (DRAM,)),
        IOConfig((PMEM, PMEM), (PMEM,)),
    ]
    @test configs == expected
end

function test_embedding()
    context_size = 2
    embedding_dim = 10
    vocab = 96
    embedding_param = Flux.param(randn(Float32, embedding_dim, length(vocab)))

    f = function(x)
        a = reshape(embedding(x, embedding_param), :)
        # Mark the embedding parameter as an inplace update node.
        nGraph.__inplace(embedding_param)
        return softmax(a)
    end

    loss(a, b) = Flux.crossentropy(f(a), b)

    # Construct inputs and output placeholders
    inputs = zeros(Int32, context_size)
    expected = randn(Float32, embedding_dim * context_size)

    backend = nGraph.Backend("CPU")
    F = nGraph.compile(
        backend,
        loss,
        inputs,
        expected;
        optimizer = nGraph.SGD(Float32(0.01))
    )
    return F
end

@testset "Test `inplace` annotation" begin
    backend = nGraph.Backend("CPU")
    cache = AutoTM.Profiler.CPUKernelCache(joinpath(@__DIR__, "cache.bin"); force_new = true)

    # Use the `test_embedding` from nGraph.jl to generate an embedding table with backprop.
    fex = test_embedding()
    data = AutoTM.Profiler.profile(fex; recache = true, cache = cache)

    # Find the embedding backprop node in the graph
    ind = findfirst(x -> nGraph.description(unx(x)) == "EmbeddingLookupBackprop", nodes(data))
    node = nodes(data)[ind] 

    println("Displaying Embedding Backprop Timings")
    display(node.timings)
    println()

    # Because of the inplace annotation - there should be only 8 entries for the 
    # 4 inputs/outputs of the node since one input/output pair are joined.
    @test length(node.timings) == 8
end
