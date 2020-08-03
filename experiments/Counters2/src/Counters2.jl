module Counters2

# stdlib
using Dates

# for ordereddict
using DataStructures

# Measurement utilities
using MattDaemon
using CounterTools
using SystemSnoop
using DataCollection
using TimeSync

# The main thing
using AutoTM
using nGraph

#####
##### Global constants
#####

socket() = CounterTools.IndexZero(1)
port() = 2000
cpus() = 25:48
sampletime() = Dates.Millisecond(500)
database() = joinpath(dirname(@__DIR__), "autotm_counters.jls")
database_snoop() = joinpath(dirname(@__DIR__), "autotm_snooped_counters.jls")

const UncoreSelectRegister = CounterTools.UncoreSelectRegister
const CoreSelectRegister = CounterTools.CoreSelectRegister

function enable_timestamps()
    # Enable timestamping of ngraph kernels
    ENV["NGRAPH_KERNEL_TIMESTAMPS"] = true
end

#####
##### Interface AutoTM stuff with DataCollection
#####

# backend
DataCollection.lower(x::nGraph.Backend{T}) where {T} = string(T)

# model constructors
const ExperimentType = Union{
    AutoTM.Experiments.Resnet,
    AutoTM.Experiments.Inception_v4,
    AutoTM.Experiments.DenseNet,
    AutoTM.Experiments.Vgg
}
DataCollection.lower(x::ExperimentType) = AutoTM.Experiments.name(x)

# optimizers
DataCollection.lower(x::AutoTM.Optimizer.AbstractOptimizer) = AutoTM.Optimizer.name(x)

# cached
DataCollection.lower(x::T) where {T <: AutoTM.Profiler.AbstractKernelCache} = string(T)

#####
##### Counter Setup
#####

# Group IMC Counters into sets of four since we can only program four counters at a time.
const GROUPED_IMC_COUNTERS = [
    (
        UncoreSelectRegister(; event = 0xD3, umask = 0x01) => "Tag Hit",
        UncoreSelectRegister(; event = 0xD3, umask = 0x02) => "Tag Miss Clean",
        UncoreSelectRegister(; event = 0xD3, umask = 0x04) => "Tag Miss Dirty",
        UncoreSelectRegister(; event = 0x00, umask = 0x00) => "Uncore Clocks",
    ),
    (
        UncoreSelectRegister(; event = 0x04, umask = 0x03) => "DRAM Read",
        UncoreSelectRegister(; event = 0x04, umask = 0x0C) => "DRAM Write",
        UncoreSelectRegister(; event = 0xEA, umask = 0x02) => "PMM Read",
        UncoreSelectRegister(; event = 0xEA, umask = 0x04) => "PMM Write",
    )
]

function coreevents()
    return (
        # Just a dummy event for now
        events = (
            CoreSelectRegister(),
        ),
        fixed_events = (
            CounterTools.RetiredInstructions,
            CounterTools.ClockUnhaltedCore,
        ),
    )
end

#####
##### Run Parameters
#####

Base.@kwdef struct RunParameters <: DataCollection.AbstractParameters
    f::ExperimentType
    optimizer::AutoTM.Optimizer.AbstractOptimizer
    backend::nGraph.Backend
    cache::Any
    mode::String
end

#####
##### Run Loop
#####

maybeunwrap(x) = x
maybeunwrap(x::Tuple) = first(x)

function run(parameters::RunParameters)
    # Step 1 - create the desired AutoTM function and run it once to warm everything up.
    fex = AutoTM.Optimizer.factory(
        parameters.backend,
        parameters.f,
        parameters.optimizer;
        cache = parameters.cache
    ) |> maybeunwrap
    @time fex()

    sampletime_parameters = DataCollection.GenericParameters(; sampletime = sampletime())

    # Now, take IMC measurements
    for event_name_pairs in GROUPED_IMC_COUNTERS
        # Reset counters to generate a timeline after data collection.
        nGraph.reset_counters(fex.ex)

        # prepare imc events
        events = first.(event_name_pairs)

        # prepare core events
        core_nt = coreevents()
        core_events = core_nt.events
        core_fixed = core_nt.fixed_events

        # construct measurement to send to the monitor
        measurements = MattDaemon.@measurements (
            pretime = SystemSnoop.Timestamp(),
            imc = CounterTools.IMCMonitor(events, socket()),
            core = CounterTools.CoreMonitor(core_events, cpus(); fixed_events = core_fixed),
            posttime = SystemSnoop.Timestamp(),
        )

        payload = MattDaemon.ServerPayload(sampletime(), measurements)
        data, _, runtime = MattDaemon.run(fex, payload, port(); sleeptime = Second(1))
        @show runtime

        # Take counter differences and aggregate counters across all IMC Channels
        counter_values = data.imc
        deltas = CounterTools.aggregate.(diff(counter_values))
        df = DataCollection.load(database())

        for (i, name) in enumerate(last.(event_name_pairs))
            # Extract the values for this counter
            this_counter = getindex.(deltas, i)
            @show name

            dc_data = DataCollection.GenericData(; counter_values = this_counter)
            countername = DataCollection.GenericParameters(; counter_name = name)

            DataCollection.addrow!(
                df,
                dc_data,
                countername,
                parameters,
                sampletime_parameters;
                cols = :union,
            )
        end

        # Save the fixed events
        core_counter_values = data.core
        deltas = CounterTools.aggregate.(diff(counter_values))

        for (i, enum) in enumerate(core_fixed)
            this_counter = getindex.(deltas, i)
            dc_data = DataCollection.GenericData(; counter_values = this_counter)
            countername = DataCollection.GenericParameters(; counter_name = string(enum))

            DataCollection.addrow!(
                df,
                dc_data,
                countername,
                parameters,
                sampletime_parameters;
                cols = :union,
                force = true,
            )
        end

        DataCollection.save(df, database())
    end

    ### Collect heap data
    record = heap_record(fex, parameters.backend)
    df = DataCollection.load(database())
    DataCollection.addrow!(
        df,
        DataCollection.GenericData(; heap_record = record),

        # Need to add this to get around a limitation in DataCollection.jl
        DataCollection.GenericParameters(; record = "heap record"),
        parameters,
        sampletime_parameters;
        cols = :union,
    )

    DataCollection.save(df, database())
    return fex
end

function heap_record(fex, backend)
    function_data = AutoTM.Profiler.FunctionData(fex.ex.ngraph_function, backend)
    times = nGraph.get_performance(fex.ex)

    # Record metadata about each tensor that will be used for plot generation.
    tensor_records = map(collect(AutoTM.Profiler.tensors(function_data))) do tensor
        users = AutoTM.Profiler.users(tensor)
        return Dict(
             "name" => nGraph.name(tensor),
             "users" => nGraph.name.(users),
             "user_indices" => getproperty.(users, :index),
             "sizeof" => sizeof(tensor),
             "offset" => Int(AutoTM.Profiler.getoffset(tensor)),
        )
    end

    # Collect metadata on each node.
    node_records = map(AutoTM.Profiler.nodes(function_data)) do node
        outputs = nGraph.name.(AutoTM.Profiler.outputs(node))
        inputs = nGraph.name.(AutoTM.Profiler.inputs(node))
        time = get(times, nGraph.name(node), 0)
        delete!(times, nGraph.name(node))
        return Dict(
            "name" => nGraph.name(node),
            "inputs" => inputs,
            "outputs" => outputs,
            "time" => time,
        )
    end

    @assert isempty(times)

    # Combine together to the final record that will be saved.
    record = Dict(
        "tensors" => tensor_records,
        "nodes" => node_records,
    )

    return record
end

#####
##### Top Level Experiments
#####

function testrun()
    parameters = RunParameters(
        f = AutoTM.Experiments.test_vgg(),
        optimizer = AutoTM.Optimizer.Optimizer2LM(),
        cache = "nocache",
        backend = nGraph.Backend("CPU"),
        mode = "2LM"
    )

    run(parameters)
end

function experiment2lm(f)
    optimizer = AutoTM.Optimizer.Optimizer2LM()
    cache = "nocache"
    backend = nGraph.Backend("CPU")

    parameters = RunParameters(
        f = f,
        optimizer = optimizer,
        cache = cache,
        backend = backend,
        mode = "2LM",
    )

    return run(parameters)
end

function experiment1lm(f; limit = 185_000_000_000)
    optimizer = AutoTM.Optimizer.Synchronous(limit)
    cache = AutoTM.Profiler.CPUKernelCache(AutoTM.Experiments.CPU_CACHE)
    backend = nGraph.Backend("CPU")

    parameters = RunParameters(
        f = f,
        optimizer = optimizer,
        cache = cache,
        backend = backend,
        mode = "1LM",
    )

    return run(parameters)
end

############################################################################################
############################################################################################
############################################################################################

# This run collects the timestamp of nGraph kernels to correlate kernel execution with
# counter values
function snooped_run(parameters::RunParameters)
    # Step 1 - create the desired AutoTM function and run it once to warm everything up.
    fex = AutoTM.Optimizer.factory(
        parameters.backend,
        parameters.f,
        parameters.optimizer;
        cache = parameters.cache
    ) |> maybeunwrap
    @time fex()

    sampletime_parameters = DataCollection.GenericParameters(; sampletime = Millisecond(100))

    # Now, take IMC measurements
    for event_name_pairs in GROUPED_IMC_COUNTERS
        # Reset counters to generate a timeline after data collection.
        nGraph.reset_counters(fex.ex)

        events = first.(event_name_pairs)
        measurements = MattDaemon.@measurements (
            pretime = SystemSnoop.Timestamp(),
            imc = CounterTools.IMCMonitor(events, socket()),
            posttime = SystemSnoop.Timestamp(),
        )

        payload = MattDaemon.ServerPayload(Millisecond(100), measurements)

        # Capture output stream to a buffer
        local data
        local runtime

        stdout_file = joinpath(dirname(@__DIR__), "stdout")
        open(stdout_file; write = true) do io
            redirect_stdout(io) do
                data, _, runtime = MattDaemon.run(fex, payload, port(); sleeptime = Second(1))
            end
        end

        # Now, associate time stamps with each kernel
        kernel_to_time = OrderedDict{String,Int}()
        open(stdout_file) do io
            # Each line has an entry
            #
            # "nodename timestamp"
            #
            # So it's pretty easy to parse
            for ln in eachline(io)
                sp = split(ln)
                nodename = String(sp[1])
                timestamp = parse(Int, sp[2])
                kernel_to_time[nodename] = timestamp
            end
        end

        # Open up the DataFrame where our data is being collected.
        df = DataCollection.load(database_snoop())

        #####
        ##### Node Time Processing
        #####

        # Adjust the c++ time to Julia time, then to a DateTime object
        jt = TimeSync.juliatime()
        ct = TimeSync.cxxtime()
        adjustment = jt - ct
        kernel_to_time = OrderedDict(
            k => unix2datetime((v + adjustment) / 1E6) for (k,v) in kernel_to_time
        )

        # Save the timings for each kernel
        node_params = DataCollection.GenericParameters(;
            counter_group = last.(event_name_pairs),
        )

        node_data = DataCollection.GenericData(;
            kernel_to_time = kernel_to_time
        )

        DataCollection.addrow!(
            df,
            node_data,
            node_params,
            parameters,
            sampletime_parameters;
            cols = :union
        )

        #####
        ##### Counter Processing
        #####

        # Take counter differences and aggregate counters across all IMC Channels
        counter_values = data.imc
        deltas = CounterTools.aggregate.(diff(counter_values))

        # Get timestamps in multiple formats for later review
        timestamps = data.pretime

        for (i, name) in enumerate(last.(event_name_pairs))
            # Extract the values for this counter
            this_counter = getindex.(deltas, i)
            @show name

            dc_data = DataCollection.GenericData(;
                counter_values = this_counter,
                timestamps = timestamps,
            )

            countername = DataCollection.GenericParameters(; counter_name = name)

            DataCollection.addrow!(
                df,
                dc_data,
                countername,
                parameters,
                sampletime_parameters;
                cols = :union,
            )

        end

        DataCollection.save(df, database_snoop())
    end

    ### Collect heap data
    record = heap_record(fex, parameters.backend)
    df = DataCollection.load(database_snoop())
    DataCollection.addrow!(
        df,
        DataCollection.GenericData(; heap_record = record),

        # Need to add this to get around a limitation in DataCollection.jl
        DataCollection.GenericParameters(; record = "heap record"),
        parameters,
        sampletime_parameters;
        cols = :union,
    )

    DataCollection.save(df, database_snoop())
    return fex
end

function experiment2lm_snooped(f)
    enable_timestamps()
    optimizer = AutoTM.Optimizer.Optimizer2LM()
    cache = "nocache"
    backend = nGraph.Backend("CPU")

    parameters = RunParameters(
        f = f,
        optimizer = optimizer,
        cache = cache,
        backend = backend,
        mode = "2LM",
    )

    return snooped_run(parameters)
end

end # module

