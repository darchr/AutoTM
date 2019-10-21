# Counters

Record hardware counter metrics for large workloads running in 1LM or 2LM modes.
Used to compare the differences between bandwidths in these two modes and to gain further
insight into the behavior of the 2LM cache.

## Setup

This experiment consists of two files that run in separate Julia processes.
First, start up the performance counter server.
In this directory, perform the following:
```
sudo <path-to-julia> counters.jl
```

In a separate console window, run either `run_1lm.sh` or `run_2lm.sh` as appropriate.
If you desire to run only a subset of these workloads, edit the run scripts.

### Two Process Reasoning

By default, all julia I/O tasks, timers etc are multiplexed through a single OS thread.
This includes calls into shared libraries.
Thus, calling into the ngraph library essentiall stops the same Julia process from
    concurrently reading hardware counters.
Julia provides a hook to get around this with the `@threadcall` macro
(https://docs.julialang.org/en/v1/manual/parallel-computing/#@threadcall-(Experimental)-1),
    but since we are using [CxxWrap](https://github.com/JuliaInterop/CxxWrap.jl), we don't
    get this option.
Therefor, two separate processes are used - one monitoring the performance counters and
    one for managing all of the ngraph stuff.
The two processes communicate through a NamedPipe.
Since we're using two processes, and performance counters need `sudo` to work correctly,
    we can run the performance counter code under `sudo` and the other code as a normal user.
