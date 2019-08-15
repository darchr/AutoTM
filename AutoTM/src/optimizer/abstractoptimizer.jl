const OPTIM_TYPES = Union{Int64, Rational{Int64}}

"""
    AbstractOptimizer{T} where {T <: Union{Int64, Rational{Int64}}}

An `AbstractOptimizer` is used to dispatch to various backend optimization routines like
static, synchronous, asynchronous, and others.

The type parameter encodes some of the high level behavior:

* `Int64`: The limit supplied by the type is an absolute memory limit.

* `Rational{Int64}`: The limit supplied by the type is a ratio of Large Memory (i.e. PMEM)
    to Limited Memory (i.e. DRAM)

    This will dispatch to another routine that will perform a grid search on the ratio to
    find an input ratio to supply that will result in an actual output ratio closest to
    the original input.
"""
abstract type AbstractOptimizer{T <: OPTIM_TYPES} end

const BANDWIDTHS = (
    # Split out CPU based on cases.
    cpu_pmem_dram_sync = 29000,
    cpu_dram_pmem_sync = 12000,
    cpu_pmem_dram_async = 2000,
    cpu_dram_pmem_async = 2500,

    # GPU Basically same in all diractions
    gpu = 12000,
)

_bw_remote_local_sync(::nGraph.Backend{nGraph.CPU}) = BANDWIDTHS[:cpu_pmem_dram_sync]
_bw_local_remote_sync(::nGraph.Backend{nGraph.CPU}) = BANDWIDTHS[:cpu_dram_pmem_sync]
_bw_remote_local_async(::nGraph.Backend{nGraph.CPU}) = BANDWIDTHS[:cpu_pmem_dram_async]
_bw_local_remote_async(::nGraph.Backend{nGraph.CPU}) = BANDWIDTHS[:cpu_dram_pmem_async]

_bw_remote_local_sync(::nGraph.Backend{nGraph.GPU}) = BANDWIDTHS[:gpu]
_bw_local_remote_sync(::nGraph.Backend{nGraph.GPU}) = BANDWIDTHS[:gpu]
_bw_remote_local_async(::nGraph.Backend{nGraph.GPU}) = BANDWIDTHS[:gpu]
_bw_local_remote_async(::nGraph.Backend{nGraph.GPU}) = BANDWIDTHS[:gpu]

# Bound on the actual between requested ratio for a workload.
#
# We iterate on the solution untill the ratio of PMEM to DRAM is within this bound of
# the requested ratio.
const RATIO_TOLERANCE = Ref(0.05)

function checkmargin(actual, wanted, tol = RATIO_TOLERANCE[])
    # Handle the cases where the denominator of "wanted" is zero
    rwanted, ractual = getratio.((wanted, actual))

    if iszero(rwanted.den) || iszero(rwanted.num)
        return true
    else
        return abs(ractual / rwanted - 1) <= tol
    end
end

geterr(actual, wanted) = abs(getratio(actual) / getratio(wanted) - 1)

Utils.getratio(x::AbstractOptimizer{Rational{Int64}}) = x.ratio
Utils.getratio(x::AbstractOptimizer{Int}) = getlimit(x)
Utils.getratio(x::Number) = x

getlimit(x::AbstractOptimizer{Int64}) = x.ratio

_optimizer(::T, r) where {T <: AbstractOptimizer} = T(r)

_numerator(x::AbstractOptimizer{Rational{Int64}}) = getratio(x).num
_numerator(x::AbstractOptimizer{Int64}) = getlimit(x)

_denominator(x::AbstractOptimizer{Rational{Int64}}) = getratio(x).den
_denominator(x::AbstractOptimizer{Int64}) = one(Int64)

name(::T) where {T <: AbstractOptimizer} = error("Name of $T not implemented")
