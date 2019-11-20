# Test `SocketCounterRecord`
@testset "SocketCounterRecord" begin
    a = Counters.SocketCounterRecord{6}()

    # Populate some fields
    a.aggregate_core[:dtlb_store_misses] = [10, 11]
    a.imc_channels[1][:tag_hits] = [20, 21]

    # Create another one and make sure `merge!` works.
    b = Counters.SocketCounterRecord{6}()
    b.aggregate_core[:dtlb_load_misses] = [20]
    b.imc_channels[1][:tag_hits] = [10]
    b.imc_channels[2][:tag_misses] = [100]

    merge!(a, b)
    @test a.aggregate_core[:dtlb_store_misses] == [10, 11]
    @test a.aggregate_core[:dtlb_load_misses] == [20]
    # Should be replaced by `b`
    @test a.imc_channels[1][:tag_hits] == [10]
    @test a.imc_channels[2][:tag_misses] == [100]

    # Create another dummy channel to make sure the non-mutating `merge` works.
    c = Counters.SocketCounterRecord{6}()
    c.aggregate_core[:entry] = [0]
    c.imc_channels[3][:hello] = [-1]

    d = merge(a, c)
    @test d.aggregate_core[:dtlb_store_misses] == [10, 11]
    @test d.aggregate_core[:dtlb_load_misses] == [20]
    @test d.aggregate_core[:entry] == [0]
    @test d.imc_channels[1][:tag_hits] == [10]
    @test d.imc_channels[2][:tag_misses] == [100]
    @test d.imc_channels[3][:hello] == [-1]
end

@testset "Testing PCM Hooks" begin
    measurements = (
        counters = Counters.CoreMonitorWrapper(),
    )

    # Make sure we get the resulting data back in the way we expect.
    trace, processed = SystemSnoop.snoop(measurements) do snooper
        for i in 1:10
            sleep(0.1)
            SystemSnoop.measure!(snooper)
        end
        return SystemSnoop.postprocess(snooper)
    end

    # Get out the socket data from the returned NamedTuple
    data = processed.socket_1
    @test isa(data, Counters.SocketCounterRecord)

    events = Counters.default_events()
    for event in events
        # Make sure the events were recorded and that they are the correct length.
        @test haskey(data.aggregate_core, event.name)
        @test length(data.aggregate_core[event.name]) == 10
    end

    # Now try Uncore Counters
    counters = Counters.DEFAULT_NT
    measurements = (
        counters = Counters.Uncore{2,2,6}(counters),
    )

    # Make sure we get the resulting data back in the way we expect.
    trace, processed = SystemSnoop.snoop(measurements) do snooper
        for i in 1:10
            sleep(0.1)
            SystemSnoop.measure!(snooper)
        end
        return SystemSnoop.postprocess(snooper)
    end
    nsockets = 2

    for i in 1:nsockets
        # Get out the socket data from the returned NamedTuple
        data = getproperty(processed, Symbol("socket_$(i-1)"))
        @test isa(data, Counters.SocketCounterRecord)

        L = length(data.imc_channels)
        for i in 1:L
            for name in keys(counters)
                @test haskey(data.imc_channels[i], name)
                @test length(data.imc_channels[i][name]) == 10
            end
        end
    end
end

