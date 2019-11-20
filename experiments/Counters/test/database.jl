# For testing the inner merging functionality of `addentry!`
struct TestMerger
    val::Int
end
Counters.datamerge(a::TestMerger, b::TestMerger) = TestMerger(a.val + b.val)

@testset "Testing DataBase" begin
    # Test ntsorter
    nt = (c = 1, b = 2, a = 3)
    @test Counters.ntsorter(nt) == (a = 3, b = 2, c = 1)

    # Test `TableEntry`
    T = Counters.TableEntry((a = 1, b = 2, c = missing), Dict(:data => "hello"))

    # Test matching when `params` just a subset of the parameters of `T`.
    @test Counters.ismatch(T, NamedTuple())      == true
    @test Counters.ismatch(T, (a = 1,))          == true
    @test Counters.ismatch(T, (b = 2,))          == true
    @test Counters.ismatch(T, (a = 1, b = 2))    == true
    @test Counters.ismatch(T, (c = missing,))    == true

    # Test matching when `params` is NOT a subset
    @test Counters.ismatch(T, (c = 1,))          == false
    @test Counters.ismatch(T, (c = 1, a = 1))    == false

    # Test `getproperty`
    @test isa(T.created, Dates.DateTime)
    @test T.data[:data] == "hello"
    @test T.a == 1
    @test T.b == 2
    @test ismissing(T.c)
    @test propertynames(T) == (:created, :updated, :data, :a, :b, :c)

    # Test the promotion properties.
    U = Counters.TableEntry((d = 0,), Dict(:UData => "bye"))

    pt = promote_type(typeof(T), typeof(U))
    @test pt == Counters.TableEntry{(:a, :b, :c, :d)}
    V = convert(pt, T)

    @test V.a == T.a
    @test V.b == T.b
    @test V.c === T.c
    @test ismissing(V.d)
    @test V.data[:data] == "hello"

    # Test conversion of `U`
    W = convert(pt, U)
    @test ismissing(W.a)
    @test ismissing(W.b)
    @test ismissing(W.c)
    @test W.d == U.d
    @test W.data[:UData] == "bye"

    # Now try some schenanigans with a DataBase.
    database = Counters.DataTable()
    @test length(database) == 0

    # Make sure showing an empty database doesn't error out.
    println(database)

    # Start adding entries to make sure that everything works more or less as we want.
    database = Counters.addentry!(database, (a = 1, b = 2), Dict(:data => TestMerger(10)))
    @test length(database) == 1
    println(database)
    @test propertynames(database) == (:created, :updated, :data, :a, :b)

    # Add data that matches
    database = Counters.addentry!(database, (a = 1, b = 2), Dict(:data => TestMerger(100), :a => 1))
    # Ensure inner `datamerge` was called correctly
    @test database.entries[1].data[:data] == TestMerger(110)
    @test database.entries[1].data[:a] == 1

    # Make sure information propogation works.
    database = Counters.addentry!(database, (b = 3, c = 4))
    @test length(database) == 2

    # Use the accessor methods with a NamedTuple to make sure we get the rows we want.
    @test length(database[(a = 1, b = 2)]) == 1
    @test length(database[(a = 1, b = 2, c = missing)]) == 1
    @test length(database[(a = missing,)]) == 1

    # Extract data using the Tables interface.
    subdatabase = database[(a = 1,)]
    cols = Tables.columns(subdatabase)
    data = getproperty(cols, :data)

    # Extract the data and make sure it propogated correctly.
    @test isa(data, Vector{Dict{Symbol,Any}})
    @test length(data) == 1
    data = first(data)
    @test data[:data] == TestMerger(110)
end
