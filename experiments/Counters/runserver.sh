#!/bin/sh

JL_PATH=~/projects/julia/julia
sudo $JL_PATH --color=yes --project -e "using Counters; Counters.runserver()"
