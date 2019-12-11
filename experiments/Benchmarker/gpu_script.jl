# Simple script for running the GPU benchmarks.
#
# Fragmentation in GPU arrays allocated by CuArrays requires the Julia executable to be
# restarted between runs.

if length(ARGS) == 1
    jl_bin = first(ARGS)
else
    jl_bin = "julia"
end

@info """
    Using "$jl_bin" for the Julia binary executable.
    Run this script as

        julia gpu_script.jl <path/to/julia>

    to override.
    """

# Go through each of the GPU benchmarks
for i in 7:11
    # Go through the optimizers
    for opt in ("Synchronous", "Asynchronous")
        # Build the command string
        exe = """
            using Benchmarker, AutoTM;
            Benchmarker.run_gpu($i; optimizers = AutoTM.Optimizer.$opt);
        """

        cmd = [
            jl_bin,
            "--project=$(pwd())",
            "--color=yes",
            "-E",
            exe
        ]
        run(`$cmd`)
    end
end
