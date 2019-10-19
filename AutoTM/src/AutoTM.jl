module AutoTM

# Setup environmental variables for nGraph
function __init__()
    setup_affinities()
    setup_codegen()
    setup_passes()
    
    # Setup PMEM if that option is enabled in nGraph.jl
    settings = nGraph.settings() 
    pmm_enabled = get(settings, "PMDK", false)
    pmm_enabled && setup_pmem()

    # just construct a CPU backend to make sure the CPU library is properly loaded
    #
    # Otherwise, some code that lives in the CPU portion of the nGraph code base will not
    # be loaded/links and we error :(
    backend = nGraph.Backend("CPU")     

    # DEPRECATRED: This is needed for a newer version of ngraph - but I reverted back
    # to the ASPLOS version of ngraph because the newer version was too unstable for
    # development.
    # # If pmm is enabled, configure the backend to use our custom allocator.
    # # pmm_enabled && nGraph.Lib.set_pmm_allocator(nGraph.getpointer(backend))
end

# stdlibs
using Dates, Random, Serialization, Statistics

# deps
using nGraph
using nGraph: TensorDescriptor, NodeDescriptor, inputs, outputs, description

using JuMP
using Gurobi
using LightGraphs
using IterTools
using ProgressMeter
using DataStructures
using Flux

# for the beautiful plotting!
using PGFPlotsX

# docstringing it up!
using DocStringExtensions

Backend(args...) = nGraph.Backend(args...)

#####
##### Core Functionality
#####

include("setup.jl")
include("utils/utils.jl")
include("profiler/profiler.jl")
include("optimizer/optimizer.jl")
include("verifier/verifier.jl")

# Bring in some utils stuff
using .Utils: Actualizer, actualize, @closure
using .Verifier: verify

#####
##### Predefined models
#####

include("zoo/zoo.jl")

#####
##### Plotting
#####

include("visualizer/visualizer.jl")

#####
##### Experiments
#####

include("experiments/experiments.jl")

end # module
