
# See `pcm/types.h - lines 746-747`
perf_event(x) = (UInt(x) << 0)
perf_umask(x) = (UInt(x) << 8)

#####
##### PCM Core
#####

# By default - all our experiments happen on Socket 2 (of 2) - only gather counter values
# for CPU 0 on each socket.
#
# `numactl` makes sure that threads run on the correct cores.
default_cpu_mask() = 25:48

# Type stable wrapping for making the return value of SystemSnoop nice.
struct CoreMonitorWrapper{names}
    monitor::PCM.CoreMonitor
end

default_events() = [
    PCM.EventDescription(0x08, 0x0E, :dtlb_load_miss),
    PCM.EventDescription(0x49, 0x0E, :dtlb_store_miss),
    PCM.EventDescription(0xD0, 0x11, :stlb_load_miss),
    PCM.EventDescription(0xD0, 0x12, :stlb_store_miss),
]

function CoreMonitorWrapper(; events = default_events(), cores = default_cpu_mask())
    monitor = PCM.CoreMonitor(; cores = cores)
    PCM.program(monitor, events)
    names = ntuple(i -> events[i].name, length(events))
    return CoreMonitorWrapper{names}(monitor)
end

# SystemSnoop API
SystemSnoop.prepare(C::CoreMonitorWrapper) = PCM.sample!(C.monitor)

function SystemSnoop.measure(C::CoreMonitorWrapper{names}) where {names}
    PCM.sample!(C.monitor)
    vals = PCM.getcounters(C.monitor)
    return NamedTuple{names}(ntuple(i -> vals[i], length(names)))
end

SystemSnoop.clean(::CoreMonitorWrapper) = PCM.cleanup()

#####
##### PCM Uncore
#####

# some common events from download.01.org
"All DRAM Read CAS Commands issued (including underfills)"
cas_count_rd() = perf_event(0x4) + perf_umask(0x3)

"All DRAM Write CAS commands issued"
cas_count_wr() = perf_event(0x4) + perf_umask(0xc)

"Regular reads(RPQ) commands for Intel Optane DC persistent memory"
pmm_read_cmd() = perf_event(0xe3)

"Write commands for Intel Optane DC persistent memory"
pmm_write_cmd() = perf_event(0xe7)

"All hits to Near Memory(DRAM cache) in Memory Mode"
tagchk_hit() = perf_event(0xd3) + perf_umask(0x1)

"All Clean line misses to Near Memory(DRAM cache) in Memory Mode"
tagchk_miss_clean() = perf_event(0xd3) + perf_umask(0x2)

"All dirty line misses to Near Memory(DRAM cache) in Memory Mode"
tagchk_miss_dirty() = perf_event(0xd3) + perf_umask(0x4)

"All commands for Intel Optane DC persistent memory"
all_pmm_cmd() = perf_event(0xea) + perf_umask(0x1)

"Clockticks of the memory controller which uses a programmable counter"
unc_clocks() = perf_event(0x0) + perf_umask(0x0)

"Read Pending Queue Occupancy of all read requests for Intel Optane DC persistent memory"
pmm_rq() = perf_event(0xe0) + perf_umask(0x1)

"Write Pending Queue Occupancy of all write requests for Intel Optane DC persistent memory"
pmm_wq() = perf_event(0xe4) + perf_umask(0x1)

dram_rq() = perf_event(0x80) + perf_umask(0x0)
dram_wq() = perf_event(0x81) + perf_umask(0x0)

"Underfill read commands for Intel Optane DC persistent memory"
pmm_underfill_rd() = perf_event(0xea) + perf_umask(0x8)

"Read requests allocated in the PMM Read Pending Queue for Intel Optane DC persistent memory"
pmm_read_insert() = perf_event(0xe3) + perf_umask(0x0)

"Write requests allocated in the PMM Write Pending Queue for Intel Optane DC persistent memory"
pmm_write_insert() = perf_event(0xe7) + perf_umask(0x0)

llc_data_read() = perf_event(0x34) + perf_umask(0x3)
llc_data_write() = perf_event(0x34) + perf_umask(0x5)

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

function SystemSnoop.prepare(U::Uncore{NS, NIMC, NCH}, kw) where {NS, NIMC, NCH}
    # Sample once to clear running counters
    PCM.sample!(U.monitor)
end

function SystemSnoop.measure(U::Uncore{NS, NIMC, NCH, names}, kw) where {NS, NIMC, NCH,names}
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
