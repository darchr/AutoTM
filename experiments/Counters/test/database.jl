# For testing the inner merging functionality of `addentry!`
struct TestMerger
    val::Int
end
Counters.datamerge(a::TestMerger, b::TestMerger) = TestMerger(a.val + b.val)

@testset "Testing Database" begin
    # Test ntsorter
    nt = (c = 1, b = 2, a = 3)
    @test Counters.ntsort(nt) == (a = 3, b = 2, c = 1)

    # ismatch
    a = (a = 1, b = 2)
    @test Counters.ismatch(a, a)

    # Make sure "data" is ignored
    a = (a = 1, b = 2, data = 10)
    b = (a = 1, b = 2, data = 5)
    @test Counters.ismatch(a, b)

    # Test missing
    a = (a = nothing, b = 2)
    b = (a = nothing, b = 2)
    @test Counters.ismatch(a, b)

    b = (a = 1, b = 2)
    @test Counters.ismatch(a, b) == false

    # Test subset
    b = (a = nothing,)
    @test Counters.ismatch(a, b)

    # Test expansion.
    @test Counters.expand((a = 1, c = 2), (:b, :c)) == (a = 1, b = nothing, c = 2)

    #####
    ##### The database
    #####
    # Now try some schenanigans with a DataBase.
    db = Counters.database()
    @test length(db) == 0

    # Make sure showing an empty database doesn't error out.
    println(db)

    # Start adding entries to make sure that everything works more or less as we want.
    db = Counters.addentry!(db, (a = 1, b = 2), Dict(:data => TestMerger(10)))
    @test length(db) == 1
    println(db)

    # Add data that matches
    db = Counters.addentry!(db, (a = 1, b = 2), Dict(:data => TestMerger(100), :a => 1))
    # Ensure inner `datamerge` was called correctly
    @test db[1].data[:data] == TestMerger(110)
    @test db[1].data[:a] == 1

    # Make sure information propogation works.
    db = Counters.addentry!(db, (b = 3, c = 4))
    @test length(db) == 2
end

