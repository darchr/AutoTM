module PCMSnoop

using PCM
using SystemSnoop

"""
* `NS`: Number of Sockets
* `NIMC`: Number of Integrated Memory Controllers per Socket
* `NCH`: Number of Memory Channels per Socket
"""
struct Uncore{NS, NIMC, NCH}
    monitor::PCM.UncoreMemoryMonitor
end

# Instantiate the controller and start the counters
function Uncore{NS, NIMC, NCH}() where {NS, NIMC, NCH}
    PCM.cleanup()

    monitor = PCM.UncoreMemoryMonitor(NIMC, NCH)
    return Uncore{NS, NIMC, NCH}(monitor)
end

# Return type from measurements for a single socket
struct UncoreMeasurements{NIMC, NCH}
    dram_reads ::NTuple{NCH,Int}
    dram_writes::NTuple{NCH,Int}
    pmm_reads  ::NTuple{NCH,Int}
    pmm_writes ::NTuple{NCH,Int}
    pmm_hitrate::NTuple{NIMC,Float64}
end

function SystemSnoop.prepare(U::Uncore{NS, NIMC, NCH}, args...) where {NS, NIMC, NCH}
    # Sample once to clear running counters
    PCM.sample!(U.monitor) 

    # Calculate the return type
    rettype = NTuple{NS, UncoreMeasurements{NIMC, NCH}}
    return Vector{rettype}()
end

function SystemSnoop.measure(U::Uncore{NS, NIMC, NCH}) where {NS, NIMC, NCH}
    PCM.sample!(U.monitor)
    return map(Tuple(1:NS)) do idx
        # Subtract 1 to get the socket         
        socket = idx - 1 
        return UncoreMeasurements{NIMC, NCH}(
            Tuple(PCM.dram_reads(U.monitor, socket)),
            Tuple(PCM.dram_writes(U.monitor, socket)),
            Tuple(PCM.pmm_reads(U.monitor, socket)),
            Tuple(PCM.pmm_writes(U.monitor, socket)),
            Tuple(PCM.pmm_hitrate(U.monitor, socket)),
        )
    end
end

SystemSnoop.clean(U::Uncore) = PCM.cleanup()

end
