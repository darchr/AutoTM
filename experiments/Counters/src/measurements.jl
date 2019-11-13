# Generator functions for measurements.
function core_events(events = default_events(), cores = default_cpu_fore_mask())
    return (
        timestamp = SystemSnoop.Timestamp(),
        counters = CoreMonitorWrapper(;events = events, cores = cores)
    )
end

function uncore_events(counters)
    return (
        timestamp = SystemSnoop.Timestamp(),
        counters = Uncore{2,2,6}(counters)
    )
end
