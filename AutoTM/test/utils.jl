# Test the "Actualize" actually does what it's supposed to.
@testset "Testing utils" begin
    backend = AutoTM.Backend("CPU")

    #####
    ##### Actualize, @closure and friends
    #####

    # Inference
    f = AutoTM.@closure AutoTM.Zoo.vgg19_inference(16)
    @test isa(f(), AutoTM.Utils.Actualizer)
    F = AutoTM.actualize(backend, f)

    @test isa(F(), nGraph.TensorView)

    # Training
    f = AutoTM.@closure AutoTM.Zoo.vgg_training(AutoTM.Zoo.Vgg19(), 16)
    @test isa(f(), AutoTM.Utils.Actualizer)
    F = AutoTM.actualize(backend, f)

    @test isa(F(), nGraph.TensorView)

    #####
    ##### findonly
    #####

    x = [2, 3, 4]
    # Examples where just one object if found
    @test AutoTM.Utils.findonly(isequal(2), x) == 1
    @test AutoTM.Utils.findonly(x -> iszero(mod(x, 3)), x) == 2
    # No items found
    @test_throws ArgumentError AutoTM.Utils.findonly(x -> iszero(x), x)
    # More than 1 item found.
    @test_throws ArgumentError AutoTM.Utils.findonly(x -> x > 0, x)

    #####
    ##### dict_push!
    #####

    d = Dict{Int,Vector{Int}}()
    AutoTM.Utils.dict_push!(d, 1, 2)
    @test d == Dict(1 => [2])

    AutoTM.Utils.dict_push!(d, 2, 2)
    @test d == Dict(1 => [2], 2 => [2])

    AutoTM.Utils.dict_push!(d, 1, 10)
    @test d == Dict(1 => [2, 10], 2 => [2])

    #####
    ##### vflatten
    #####

    x = [1,2]
    y = [3,4]
    @test collect(AutoTM.Utils.vflatten(x, y)) == [1,2,3,4]
end

@testset "Testion IOConfig" begin
    DRAM = AutoTM.Utils.DRAM
    PMEM = AutoTM.Utils.PMEM

    io = AutoTM.Utils.IOConfig((DRAM, PMEM), (PMEM, DRAM))

    @test length(io) == 4
    @test collect(io) == [DRAM, PMEM, PMEM, DRAM]
    @test io[1] == DRAM
    @test io[3] == PMEM
    @test_throws BoundsError io[0]
    @test_throws BoundsError io[5]

    # Test `isless` functionality.
    @test io < AutoTM.Utils.IOConfig((DRAM, PMEM), (PMEM, PMEM))
    @test io > AutoTM.Utils.IOConfig((DRAM, DRAM), (PMEM, PMEM))

    # Test `setindex`
    @test Base.setindex(io, 2, DRAM) == AutoTM.Utils.IOConfig((DRAM, DRAM), (PMEM, DRAM))
    @test Base.setindex(io, 4, PMEM) == AutoTM.Utils.IOConfig((DRAM, PMEM), (PMEM, PMEM))
end

@testset "Testing metagraph" begin
    # Create a dummy graph like this:
    #
    # 1 --> 2 --> 3
    #       ^     |
    #       |     |
    #       *-----*
    #
    # Associate metadata like this
    # Vertex:
    #   1 -- 'a'
    #   2 -- 'b'
    #   3 -- 'c'
    #
    # Edge:
    #   1 => 2 -- 1
    #   2 => 3 -- 2
    #   3 => 2 -- 3

    LG = AutoTM.Utils.LightGraphs
    AU = AutoTM.Utils

    mg = AU.MetaGraph(LG.SimpleDiGraph(),Int,Char)
    # Add the vertices with their metadata
    LG.add_vertex!(mg, 'a')
    LG.add_vertex!(mg, 'b')
    LG.add_vertex!(mg, 'c')

    @test AU.getmeta(mg, 1) == 'a'
    @test AU.getmeta(mg, 2) == 'b'
    @test AU.getmeta(mg, 3) == 'c'

    LG.add_edge!(mg, 1, 2, 1)
    LG.add_edge!(mg, 2, 3, 2)
    LG.add_edge!(mg, 3, 2, 3)

    @test AU.getmeta(mg, LG.edgetype(mg)(1, 2)) == 1
    @test AU.getmeta(mg, LG.edgetype(mg)(2, 3)) == 2
    @test AU.getmeta(mg, LG.edgetype(mg)(3, 2)) == 3

    # Test some of the standard lightgraphs functions to make sure things are forwarding
    # properly.
    @test LG.ne(mg) == 3
    @test LG.nv(mg) == 3
    @test LG.outneighbors(mg, 1) == [2]
    @test LG.inneighbors(mg, 2) == [1, 3]

    et = LG.edgetype(mg)
    @test Set(AU.inedges(mg, 2)) == Set((et(1, 2), et(3, 2)))
    @test Set(AU.outedges(mg, 2)) == Set((et(2, 3),))

    # Check some of the find functions.
    v = AU.find_vertex((g,v) -> AU.getmeta(g, v) == 'a', mg)
    @test v == 1
    @test AU.getmeta(mg, v) == 'a'

    # Check when number of matches is greater than 1
    @test_throws ArgumentError AU.find_vertex((g,v) -> true, mg)

    # Check when number of matches is 0
    @test_throws ArgumentError AU.find_vertex((g,v) -> false, mg)

    # Do the same thing for edges.
    e = AU.find_edge((g,v) -> AU.getmeta(g, v) == 3, mg)
    @test e == et(3, 2)
    @test_throws ArgumentError AU.find_edge((g,v) -> true, mg)
    @test_throws ArgumentError AU.find_edge((g,v) -> false, mg)
end

@testset "Testing Allocator" begin
    # Get to the AllocatorModel ... Module ...
    AllocatorModel = AutoTM.Utils.AllocatorModel

    # Instantiate a memory allocator with a memory limit of 100 and alignment of 10
    M = AllocatorModel.MemoryAllocator(100, 10)
    node_list = M.node_list
    @test length(node_list) == 1

    # Allocate a block of size 1. We expect it to be located at the beginning of the
    # block.
    offset = AllocatorModel.allocate(M, 1)
    @test offset == 0
    @test length(node_list) == 2

    # The first block should have size 10 due to alignment
    @test AllocatorModel.isfree(node_list[1]) == false
    @test sizeof(node_list[1]) == 10
    @test AllocatorModel.isfree(node_list[2]) == true
    @test sizeof(node_list[2]) == 90

    # Try freeing this block - make sure we get just a single block back
    AllocatorModel.free(M, offset)

    @test length(node_list) == 1
    @test AllocatorModel.isfree(node_list[1]) == true
    @test sizeof(node_list[1]) == 100

    # Allocate three blocks - free the middle one, then free the last
    a = AllocatorModel.allocate(M, 11)
    b = AllocatorModel.allocate(M, 9)
    c = AllocatorModel.allocate(M, 20)

    # Consistency Checks
    @test a == 0
    @test b == 20
    @test c == 30
    @test length(node_list) == 4
    @test AllocatorModel.isfree(node_list[1]) == false
    @test AllocatorModel.isfree(node_list[2]) == false
    @test AllocatorModel.isfree(node_list[3]) == false
    @test AllocatorModel.isfree(node_list[4]) == true
    @test sizeof(node_list[1]) == 20
    @test sizeof(node_list[2]) == 10
    @test sizeof(node_list[3]) == 20
    @test sizeof(node_list[4]) == 50

    AllocatorModel.free(M, b)
    @test length(node_list) == 4
    @test AllocatorModel.isfree(node_list[1]) == false
    @test AllocatorModel.isfree(node_list[2]) == true
    @test AllocatorModel.isfree(node_list[3]) == false
    @test AllocatorModel.isfree(node_list[4]) == true
    @test sizeof(node_list[1]) == 20
    @test sizeof(node_list[2]) == 10
    @test sizeof(node_list[3]) == 20
    @test sizeof(node_list[4]) == 50

    AllocatorModel.free(M, c)
    @test length(node_list) == 2
    @test AllocatorModel.isfree(node_list[1]) == false
    @test AllocatorModel.isfree(node_list[2]) == true
    @test sizeof(node_list[1]) == 20
    @test sizeof(node_list[2]) == 80

    # Test that an oversized allocation fails
    @test isnothing(AllocatorModel.allocate(M, 200))
end
