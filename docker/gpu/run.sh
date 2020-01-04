#!/bin/bash

docker run --runtime=nvidia --rm -it \
    -v "${PWD}/caches":"/AutoTM/data/caches" \
    -v "${PWD}/experiment-data":"/AutoTM/experiments/Benchmarker/data" \
    -v "${PWD}/experiment-figures":"/AutoTM/experiments/Benchmarker/figures" \
    darchr/autotm-gpu

# Experiment for tricking Gurobi into working in a Docker container
#
# Unfortunately, the license file is too smart for now ...

#GUROBI_HOME=${1:-"${PWD}/dummy-gurobi"}

# docker run --runtime=nvidia --rm -it \
#     -v "${PWD}/caches":"/AutoTM/data/caches" \
#     -v "${PWD}/experiment-data":"/AutoTM/experiments/Benchmarker/data" \
#     -v "${PWD}/experiment-figures":"/AutoTM/experiments/Benchmarker/figures" \
#     -v "/home/$(whoami)/gurobi.lic":"/home/$(whoami)/gurobi.lic" \
#     -v "$GUROBI_HOME":"/home/$(whoami)/gurobi" \
#     darchr/autotm-gpu
