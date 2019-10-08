module Snooper

# stdlib
using Dates

using PCM
using SystemSnoop

# See `pcm/types.h - lines 746-747`
perf_event(x) = (UInt(x) << 0)
perf_umask(x) = (UInt(x) << 8)

# some common events from download.01.org
cas_count_rd() = perf_event(0x4) + perf_umask(0x3)
cas_count_wr() = perf_event(0x4) + perf_umask(0xc)
pmm_read_cmd() = perf_event(0xe3)
pmm_write_cmd() = perf_event(0xe7)

#pmm_read_cmd() = perf_event(0xea) + perf_umask(0x2)
#pmm_write_cmd() = perf_event(0xea) + perf_umask(0x4)

makevar(i) = "PCM_COUNTER_$(i-1)"
function setvars(nt::NamedTuple)
    # Iterate through the named tuple - programming!
    for (i, v) in enumerate(nt)
        ENV[makevar(i)] = string(v, base = 16)
    end
end

"""
* `NS`: Number of Sockets
* `NIMC`: Number of Integrated Memory Controllers per Socket
* `NCH`: Number of Memory Channels per Socket
"""
struct Uncore{NS, NIMC, NCH, names}
    monitor::PCM.UncoreMemoryMonitor
end

# Instantiate the controller and start the counters
function Uncore{NS, NIMC, NCH}(counters::NamedTuple{names}) where {NS, NIMC, NCH, names}
    PCM.cleanup()

    # program the counters
    setvars(counters)

    monitor = PCM.UncoreMemoryMonitor(NIMC, NCH)
    return Uncore{NS, NIMC, NCH, names}(monitor)
end

# Return type from measurements for a single socket
const _NT{NCH} = NTuple{NCH,Int}
function uncore_measurements(::Uncore{NS,NIMC,NCH,names}) where {NS,NIMC,NCH,names}
    return NamedTuple{names, NTuple{4,_NT{NCH}}}
end

function SystemSnoop.prepare(U::Uncore{NS, NIMC, NCH}, args...) where {NS, NIMC, NCH}
    # Sample once to clear running counters
    PCM.sample!(U.monitor) 

    # Calculate the return type
    rettype = NTuple{NS, uncore_measurements(U)}
    return Vector{rettype}()
end

function SystemSnoop.measure(U::Uncore{NS, NIMC, NCH, names}) where {NS, NIMC, NCH,names}
    PCM.sample!(U.monitor)
    return map(Tuple(1:NS)) do idx
        # Subtract 1 to get the socket         
        socket = idx - 1 
        return NamedTuple{names}(
            ntuple(i -> Tuple(PCM.getcounter(U.monitor, i-1, socket)), 4)
        )
    end
end

SystemSnoop.clean(U::Uncore) = PCM.cleanup()

const DEFAULT_NT = (
    dram_reads  = cas_count_rd(),
    dram_writes = cas_count_wr(),
    pmm_reads   = pmm_read_cmd(),
    pmm_writes  = pmm_write_cmd(),
)

function go(pid, sampletime = 10; nt = DEFAULT_NT)

    measurements = (
        timestamp = SystemSnoop.Timestamp(),
        counters = Uncore{2,2,6}(nt)
    )

    # Catch interrupts for now so we can manually abort.
    sampler = SystemSnoop.SmartSample(Second(sampletime))
    data = SystemSnoop.snoop(SystemSnoop.SnoopedProcess(pid), measurements) do snooper
        try
            while true
                sleep(sampler)
                measure(snooper) || break
            end
        catch err
            if !isa(err, InterruptException)
                rethrow(err)
            end
        end
        return snooper.trace
    end
    return data
end

end # module
