DIR = @__DIR__
REPODIR = dirname(@__DIR__)

TEMPDIR = joinpath(DIR, "autotm_backup")
TARBALL = joinpath(DIR, "autotm_backup.tar.gz")

# Directories to save.
dirs = [
    # Kernel timing caches
    joinpath(REPODIR, "data") => "top_data",

    # Serialized results from the benchmarking runs
    joinpath(REPODIR, "experiments", "Benchmarker", "data") => "benchmark_data",
    # Figures generated from benchmarks
    joinpath(REPODIR, "experiments", "Benchmarker", "figures") => "benchmark_figures",

    # Serialized data from the performance counters runs
    joinpath(REPODIR, "experiments", "Counters", "data") => "counters_data",
    # Figures from the performance counters
    joinpath(REPODIR, "experiments", "Counters", "figures") => "counters_figures",
    # Temporary data generated from the counters runs.
    #joinpath(REPODIR, "experiments", "counters", "serialized") => "counters_serialized",
]

arg = first(ARGS)
if !in(arg, ("save", "load"))
    println("Only accepting arguments \"save\" or \"load\". Got \"$arg\"")
    exit()
end

# If we're saving, gather all of the required folders into a temporary folder and generate
# a tarball.
if arg == "save"
    # Cleanup any existing data from any previous runs
    rm(TEMPDIR; force = true, recursive = true)
    mkdir(TEMPDIR)
    for (src, dst) in dirs
        ispath(src) && cp(src, joinpath(TEMPDIR, dst))
    end
    # Generate a tarball
    run(`tar -czvf $TARBALL $(basename(TEMPDIR))`)

    # Cleanup
    rm(TEMPDIR; force = true, recursive = true)
    exit()
end

if arg == "load"
    # Do some more processing on extra arguments

    # Find if a source path is given for the tarball. If so, use it.
    # Otherwise, use the generic path for the tarball.
    ind = findlast(isequal("--tarball"), ARGS)
    if isnothing(ind)
        tarball = TARBALL
    else
        tarball = ARGS[ind + 1]
        println("Unpacking $tarball")
    end

    # Check if we should force copy the unpacked data
    ind = findfirst(isequal("--force"), ARGS)
    force = isnothing(ind) ? false : true

    # Remove an existing tempdir and unpack the tarball
    rm(TEMPDIR; force = true, recursive = true)
    run(`tar -xvf $tarball -C $(DIR)`)

    # Reverse the source and destination
    for (dst, src) in dirs
        srcpath = joinpath(TEMPDIR, src)
        ispath(srcpath) || continue

        # Check if we're forcing a copy. If not, check the dest directory and print
        # a warning if it exists
        proceed = true
        if force == false && isdir(dst)
            printstyled("Not copying to $dst. Add argument \"--force\" to override.\n"; color = :red)
            proceed = false
        end

        if proceed
            cp(joinpath(TEMPDIR, src), dst; force = force)
        end
    end

    # Cleanup
    rm(TEMPDIR; force = true, recursive = true)
    exit()
end
