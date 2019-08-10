#####
##### Methods for dealing with PMEM to DRAM ratios
#####

footprint(datum::Dict) = convert(Int, datum[:pmem_alloc_size] + datum[:dram_alloc_size])
footprint(f::nGraph.NFunction) = 
    convert(Int, nGraph.get_pmem_pool_size(f) + nGraph.get_temporary_pool_size(f))

function getratio(datum::Dict)
    r = get(datum, :ratio, nothing)
    isnothing(r) || return r

    pmem = convert(Int, datum[:pmem_alloc_size])
    dram = convert(Int, datum[:dram_alloc_size])
    return pmem // dram
end

getratio(fex::nGraph.FluxExecutable) = getratio(fex.ex.ngraph_function)
function getratio(f::nGraph.NFunction)
    pmem = convert(Int, nGraph.get_pmem_pool_size(f))
    dram = convert(Int, nGraph.get_temporary_pool_size(f))
    return pmem // dram
end

ratio_string(x::Rational) = "$(x.num):$(x.den)"
compare_ratio(a, b) = iszero(b.den) ? inv(a) : a - b

