# Instructions for use:
#
# Start julia with
#     sudo ~/julia-1.2.0/bin/julia -p1
#
# We want to start with two processes. The master process will do all the AutoTM stuff while
# the second process will be responsible for monitoring the performance counters.
#
# -- We must use two processes because we need preemptive behavior to sample the performance
#    counters while the ngraph function is running.
#
# -- We must run under `sudo` to have access to the performance counters.
#
# The script will run through a collection of AutoTM experiments to run, and before running
# will invoke a job on the second process to start sampling.
#
# The master process will write to a RemoteChannel when the ngraph function has completed -
# at which point the secondary process will stop sampling and serialize its sampled data
# to a filepath provided by the master worker.

#####
##### Setup
#####

# Select which set of tests to run
mode = "test"
#mode = "2lm"
#mode = "1lm"

sampletime = 1      # Seconds

#####
##### Code Loading
#####

# Activate the AutoTM environment, which will load all the correct versions of dependencies
@everywhere using Pkg
@everywhere Pkg.activate("../AutoTM")
@everywhere using Dates
@everywhere using AutoTM

# Get the namespace for `Snooper` into all processes
@everywhere using Snooper
@everywhere using SystemSnoop
@everywhere using Serialization

# This is the sampling function that will run on the secondary worker
@everywhere function sample(
        rr::RemoteChannel, 
        sampletime::Integer, 
        filepath::String,
        nt = Snooper.DEFAULT_NT,
    )

    # Create a measurements object from SystemSnoop
    measurements = (
        timestamp = SystemSnoop.Timestamp(),
        counters = Snooper.Uncore{2,2,6}(nt)
    )

    # Sample time regularly.
    sampler = SystemSnoop.SmartSample(Second(sampletime))

    # Normally, SystemSnoop requires a PID for what it's snooping. I no longer like this API,
    # but I guess I kind of have to deal with it for now until I get around to changing it.
    #
    # The PID is not needed for this snooping since we're monitoring the entire system,
    # so just pass the current PID
    data = SystemSnoop.snoop(SystemSnoop.SnoopedProcess(getpid()), measurements) do snooper
        while true
            # Sleep until it's time to sample.
            sleep(sampler) 

            # If something goes wrong during measurement, `measure` will return `false`.
            # We handle that gracefully by performing an early exit.
            measure(snooper) || break

            # Check to see if something has been written to the `RemoteChannel`.
            # If so, it's a sign that we must exit.
            if isready(rr) 
                # Remote the item from the channel
                take!(rr)
                break
            end
        end
        return snooper.trace
    end

    # Serialize the data to the given filepath and return a value to indicate to the
    # master process that we've completed.
    serialize(filepath, data)
    return true
end

#####
##### Here's the main sampling script.
#####

if mode == "test"
    trials = [
        (AutoTM.Experiments.test_vgg(), "test_vgg.jls")
    ]
elseif mode == "2lm"
    trials = [
        (AutoTM.Experiments.large_inception(),  "inception_2lm.jls"),
        (AutoTM.Experiments.large_vgg(),        "vgg_2lm.jls"),
        (AutoTM.Experiments.large_resnet(),     "resnet_2lm.jls"),
        (AutoTM.Experiments.large_densenet()(), "densenet_2lm.jls"),
    ]
else
    throw(error("Don't under stand mode $mode"))
end

# Try running things - see if it works
backend = AutoTM.Backend("CPU")
for trial in trials
    f = trial[1]
    filepath = trial[2]

    # Instantiate the ngraph function to run.
    fex = AutoTM.Profiler.actualize(backend, f) 

    # Run the function once to warm it up.
    @time fex() 

    # Create a RemoteChannel for communicating with the subprocess
    rr = RemoteChannel(() -> Channel{Int}(32)) 

    # Invoke the sampling subprocess
    println("Invoking Sampler")
    sampling_process = @spawnat 2 sample(rr, sampletime, filepath)

    # Sleep for 5 seconds to give the subprocess time to startup and invoke the 
    # FluxExecutable again.
    sleep(5)
    @time fex() 
    sleep(5)

    # Signal worker process that we're done by writing to the RemoteChannel
    put!(rr, 0) 
    fetch(sampling_process)
    println("Successfully Ran Worker Process")
end
