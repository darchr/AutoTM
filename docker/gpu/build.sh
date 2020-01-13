#!/bin/bash

# Check if a "gurobi" directory exists - if so, use it.
docker build -t "darchr/autotm-gpu" . \
    --build-arg USER_ID=$(id -u ${USER}) \
    --build-arg GROUP_ID=$(id -g ${USER}) \
    --build-arg WHOAMI=$(whoami)
