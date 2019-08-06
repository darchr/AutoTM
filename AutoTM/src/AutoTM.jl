module AutoTM

# Setup environmental variables for nGraph
function __init__()
    setup_affinities()
    setup_profiling()
    setup_passes()
    
    # Setup PMEM if that option is enabled in nGraph.jl
    settings = nGraph.settings() 
    get(settings, "PMDK", false) && setup_pmem()
end

# stdlibs
using Dates, Random, Serialization, Statistics

# deps
using nGraph

# Import some names
import nGraph: TensorDescriptor, NodeDescriptor, inputs, outputs, description

using JuMP, Gurobi
using LightGraphs
using IterTools
using ProgressMeter
using DataStructures
using Flux
using JSON

# for the beautiful plotting!
using PGFPlotsX

Backend(args...) = nGraph.Backend(args...)

#####
##### Core Functionality
#####
include("setup.jl")
include("utils/utils.jl")
include("profiler/profiler.jl")
include("optimizer/optimizer.jl")

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
