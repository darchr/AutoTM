module Profiler

# profiledata.jl
export  ProfileData,
        get_enums, get_times, get_time, get_bytes,
        can_select_algo,
        nodes, tensors,
        gettime,
        hastime,
        _producer,
        _consumer,
        _users,
        locations,
        get_configs,
        getconfig,
        _setup!,
        _cleanup!,
        configs_for,
        live_tensors

# cache.jl
export CPUKernelCache, GPUKernelCache

# kernels.jl
export profile, read_timing_data

# compare.jl
export compare

# stdlib deps
using Serialization

using ..Utils
using ProgressMeter
using JSON
import nGraph
import ..AutoTM._env_context

include("profiledata.jl")
include("cache.jl")
include("kernels.jl")
include("compare.jl")

end
