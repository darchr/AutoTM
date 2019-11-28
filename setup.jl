import REPL
using REPL.TerminalMenus

# Make sure JSON is installed
try
    using JSON
catch ex
    # Ask to install JSON
    choice = request(
        "Package JSON not installed - would you like me to install it?",
        RadioMenu(["yes", "no"])
    )
    if choice == 1
        using Pkg
        Pkg.add("JSON")
        using JSON
    else
        println("Canceling!")
        exit()
    end
end

# Constants for paths
AUTOTM_BUILD = joinpath(@__DIR__, "AutoTM", "deps", "build.json")
NGRAPH_BUILD = joinpath(@__DIR__, "deps", "nGraph", "deps", "build.json")

#####
##### Setup for nGraph
#####

options = ["yes", "no"]

# Select NVDIMMs
choice = request(
                 "Would you like to use NVDIMMs? (requires a Cascade Lake system with Optane)",
    RadioMenu(options)
)

# Choice for using NVDIMMs
use_nvdimms = choice == 1

# Select GPU
choice = request(
    "Would you like to build with GPU support?",
    RadioMenu(options)
)

# Choice for using NVDIMMs
use_gpu = choice == 1

ngraph_json = JSON.parsefile(NGRAPH_BUILD)

if use_nvdimms
    ngraph_json["PMDK"] = true
    ngraph_json["NUMA"] = true
else
    ngraph_json["PMDK"] = false
    ngraph_json["NUMA"] = false
end

if use_gpu
    ngraph_json["GPU"] = true
else
    ngraph_json["GPU"] = false
end

open(NGRAPH_BUILD; write = true) do f
    JSON.print(f, ngraph_json)
end

#####
##### Setup AutoTM
#####

# Select GPU
choice = request(
    "Would you like to use Gurobi as the ILP solver?",
    RadioMenu(options)
)

# Choice for using NVDIMMs
use_gurobi = choice == 1

autotm_json = JSON.parsefile(AUTOTM_BUILD)

if use_gurobi
    autotm_json["GUROBI"] = true
else
    autotm_json["GUROBI"] = false
end

open(AUTOTM_BUILD; write = true) do f
    JSON.print(f, autotm_json)
end
