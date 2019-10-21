# This file is responsible for running the counter gathering mechanism.
#
# If will watch a named pipe for instructions from a master process on when to begin
# sampling and when to end.
using Pkg; Pkg.activate("../../AutoTM")

using Sockets
using Snooper
using SystemSnoop
using Dates
using Serialization

function sample(sock, sampletime, filepath, counter_tuple)
    # Create a measurements object from SystemSnoop
    measurements = (
        timestamp = SystemSnoop.Timestamp(),
        counters = Snooper.Uncore{2,2,6}(counter_tuple)
    )

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

        data = SystemSnoop.snoop(SystemSnoop.SnoopedProcess(getpid()), measurements) do snooper
            while true
                # Sleep until it's time to sample.
                sleep(sampler)

                # If something goes wrong during measurement, `measure` will return `false`.
                # We handle that gracefully by performing an early exit.
                measure(snooper) || break

                # See if we have any communication from the master process. If so, pull it
                # out and make sure it's a `stop` command.
                if canexit
                    println("Stopping Sampling")
                    break
                end
            end
            return snooper.trace
        end
    end

    # Serialize the data to the given filepath and return a value to indicate to the
    # master process that we've completed.
    serialize(filepath, data)
    return true
end


# Create a NamedPipe to listen for commands from the other process.
server = listen("counter_pipe")

# Make readable and writeable by non-sudo processes
chmod("counter_pipe", 0x666)

while true
    sock = accept(server)

    # Setup defaults
    sampletime = 1
    filepath = "test.jls"
    counter_tuple = Snooper.DEFAULT_NT

    # Expand commands to be of the form
    # `keyword payload`
    #
    # For example, a command changing the sampletime to 2 seconds would look like
    # `sampletime 2`
    #
    # Very simple and stupid interface.
    while isopen(sock)
        cmd = readline(sock)
        if cmd == "start"
            sample(sock, sampletime, filepath, counter_tuple)

        # Key: "sampletime"
        # Payload: Integer numer of seconds
        elseif startswith(cmd, "sampletime")
            sampletime = parse(Int, last(split(cmd)))
            println("sampletime = $sampletime")

        # Key: "filepath"
        # Payload: string for a filepath
        elseif startswith(cmd, "filepath")
            filepath = last(split(cmd))
            println("filepath = $filepath")

        # Key: counters
        # Payload options:
        #   "rw" - read and write counters
        #   "tags" - cache tag counters
        elseif startswith(cmd, "counters")
            payload = last(split(cmd))
            if payload == "rw"
                counter_tuple = Snooper.DEFAULT_NT
            elseif payload == "tags"
                counter_tuple = (
                    tag_hit = Snooper.tagchk_hit(),
                    tag_miss_clean = Snooper.tagchk_miss_clean(),
                    tag_miss_dirty = Snooper.tagchk_miss_dirty(),
                    all_pmm_cmd = Snooper.all_pmm_cmd(),
                )
            elseif payload == "queues"
                counter_tuple = (
                    unc_clocks = Snooper.unc_clocks(),
                    pmm_rq = Snooper.pmm_rw(),
                    pmm_wq = Snooper.pmm_wq(),
                    all_pmm_cmd = Snooper.all_pmm_cmd(),
                )
            else
                println("Unknown Counter Payload: $payload")
            end

        # Break out of loop
        elseif cmd == "exit"
            println("Exiting")
            exit()

        # Generic error message
        else
            println("Unhandled Command: $cmd")
        end
    end
end
