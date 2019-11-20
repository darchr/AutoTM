# Run this in a separate process (with sudo privileges) to gather performance counter data.
function runserver()
    # Create a NamedPipe to listen for commands from the other process.
    server = listen(PIPEPATH)

    # Make readable and writeable by non-sudo processes
    chmod(PIPEPATH, 0x666)

    # Set up
    while true
        sock = accept(server)

        # Setup defaults
        sampletime = 1
        filepath = "test.jls"
        measurements = NamedTuple()
        params = NamedTuple()

        # Listen on the socket for instructions.
        while isopen(sock)
            cmd = readline(sock)

            # Shutdown command
            if cmd == "shutdown"
                return nothing

            # Set filepath
            elseif startswith(cmd, "filepath")
                filepath = last(split(cmd))
                println("filepath = $filepath")

            # Update `measurements` tuple.
            elseif cmd == "measurements"
                measurements = getmeasurements()
                println("measurements = $measurements")

            # Update `params` tuple
            elseif cmd == "params"
                params = getparams()
                println("params = $params")

            # Update `sampletime`.
            elseif startswith(cmd, "sampletime")
                sampletime = parse(Float64, last(split(cmd)))
                println("sampletime = $sampletime")

            # start sampling.
            elseif cmd == "start"
                GC.gc()
                sample(sock, sampletime, filepath, params, measurements)
                GC.gc()

            # who knows what happened.
            else
                println("Unknown Command: $cmd")
            end
        end
    end
end

# Deserialize the contents at `path`. It should be a function that can be called to get
# the requested measurements `NamedTuple`.
#
# The reason this has to be a function instead of just the NamedTuple itself is because the
# NamedTuple can contain some non-serializable structs.
function getmeasurements()
    x = deserialize(TRANSFERPATH)
    nt = x()::NamedTuple
    return nt
end

getparams() = deserialize(PARAMPATH)::NamedTuple

# Sampline routine.
function sample(sock, sampletime, filepath, params, measurements)
    # Sample time regularly.
    sampler = SystemSnoop.SmartSample(Second(sampletime))

    # Normally, SystemSnoop requires a PID for what it's snooping. I no longer like this API,
    # but I guess I kind of have to deal with it for now until I get around to changing it.
    #
    # The PID is not needed for this snooping since we're monitoring the entire system,
    # so just pass the current PID
    local data
    @sync begin
        # Spawn a task to sample the buffer and notify when a `stop` command is reached
        canexit = false
        @async begin
            while true
                ln = readline(sock)
                if ln == "stop"
                    canexit = true
                    break
                else
                    println("Unhandled command: $ln")
                end
            end
        end

        # Measuring loop.
        trace, data = SystemSnoop.snoop(measurements) do snooper
            while true
                # Sleep until it's time to sample.
                sleep(sampler)
                SystemSnoop.measure!(snooper)

                # See if we have any communication from the master process. If so, pull it
                # out and make sure it's a `stop` command.
                if canexit
                    println("Stopping Sampling")
                    return SystemSnoop.postprocess(snooper)
                end
            end
        end
    end

    # Serialize the data to the given filepath.
    # If the path is already valid, than first deserialize it, add our data to it, then
    # reserialize it.
    dir = dirname(filepath)
    !ispath(dir) && mkpath(dir)
    if ispath(filepath)
        x = deserialize(filepath)
    else
        x = DataTable()
    end

    # Only save the post-processed data for now.
    dict = Dict(k => v for (k,v) in pairs(data))

    # # Convert our data into to a Dict{Symbol, Any}
    # sa = StructArray(data)
    # dict = Dict(k => getproperty(sa, k) for k in getnames(eltype(sa)))

    # Merge this data in with the rest
    x = addentry!(x, params, dict)
    serialize(filepath, x)
    return true
end

