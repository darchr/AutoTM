# Collection of the actual benchmarks that are run.

"""
Run the set of benchmarks for sequential and random array access for arrays that fit within
the DRAM cache in 2LM Mode.
"""
function run_2lm_fit_in_dram()
    @assert Threads.nthreads() == 24

    # Define closures to pass to to the server.
    measurements = [
        # Only really interested in the TLB misses
        #
        # Restrict counters to cores 25:48 - the cores we're actually using.
        () -> core_events([
            PCM.EventDescription(0x08, 0x0E, :dtlb_load_miss),
            PCM.EventDescription(0x49, 0x0E, :dtlb_store_miss),
        ], 25:48),
        # Chunk up `uncore` events into groups of 4.
        #
        # First collect data on tag statistics
        () -> uncore_events((
            tag_hit         = tagchk_hit(),
            tag_miss_clean  = tagchk_miss_clean(),
            tag_miss_dirty  = tagchk_miss_dirty(),
            dram_rq         = dram_rq(),
        )),
        # Bandwidth statistics
        () -> uncore_events((
            dram_reads  = cas_count_rd(),
            dram_writes = cas_count_wr(),
            pmm_reads   = pmm_read_cmd(),
            pmm_writes  = pmm_write_cmd(),
        )),
        # Queue Depths
        () -> uncore_events((
            unc_clocks  = unc_clocks(),
            pmm_rq      = pmm_rq(),
            pmm_wq      = pmm_wq(),
            dram_wq     = dram_wq(),
        ))
    ]

    params = KernelParams(
        29,     # Around 2G per threads (2^29 * 4 B). Total array size ~51 GB.
        1.0,    # Sample time: 1 second
        "2lm",  # Run this in 2LM
        10,     # 10 inner iterations - each iteration is pretty quick.
        measurements,
        [16],       # Only use AvX 512
        [true, false],    # Use standard loads/stores
        "mmap-1gb"
    )

    runkernels(
        params,
        sz -> hugepage_mmap(Float32, sz, Pagesize1G()),
    )
end

"""
Run the set of benchmarks for sequential and random array access for arrays that do not fit
within the DRAM cache in 2LM Mode.
"""
function run_2lm_exceeds_dram()
    @assert Threads.nthreads() == 24

    # Define closures to pass to to the server.
    measurements = [
        # Only really interested in the TLB misses
        #
        # Restrict counters to cores 25:48 - the cores we're actually using.
        () -> core_events([
            PCM.EventDescription(0x08, 0x0E, :dtlb_load_miss),
            PCM.EventDescription(0x49, 0x0E, :dtlb_store_miss),
        ], 25:48),
        # Chunk up `uncore` events into groups of 4.
        #
        # First collect data on tag statistics
        () -> uncore_events((
            tag_hit         = tagchk_hit(),
            tag_miss_clean  = tagchk_miss_clean(),
            tag_miss_dirty  = tagchk_miss_dirty(),
            dram_rq         = dram_rq(),
        )),
        # Bandwidth statistics
        () -> uncore_events((
            dram_reads  = cas_count_rd(),
            dram_writes = cas_count_wr(),
            pmm_reads   = pmm_read_cmd(),
            pmm_writes  = pmm_write_cmd(),
        )),
        # Queue Depths
        () -> uncore_events((
            unc_clocks  = unc_clocks(),
            pmm_rq      = pmm_rq(),
            pmm_wq      = pmm_wq(),
            dram_wq     = dram_wq(),
        ))
    ]

    params = KernelParams(
        32,     # Around 17G per threads (2^32 * 4 B). Total array size ~413 GB.
        1.0,    # Sample time: 1 second
        "2lm",  # Run this in 2LM
        2,      # 2 iterations - these take quite a while.
        measurements,
        [16],       # Only use AvX 512
        [false, true],
        "mmap-1gb"
    )

    runkernels(
        params,
        sz -> hugepage_mmap(Float32, sz, Pagesize1G()),
    )
end

function pmm_direct_test()
    # Get the number of threads.
    # Pick the array size based on the number of workers.
    nthreads = Threads.nthreads()

    # Minimum total array size of 30 GB
    min_total_size = 30E9 / sizeof(Float32)

    # Calculate a nice power of 2 that will get us there
    array_size = convert(Int, log2(nextpow(2, ceil(Int, min_total_size / nthreads))))

    measurements = [
        # Bandwidth statistics
        () -> uncore_events((
            dram_reads  = cas_count_rd(),
            dram_writes = cas_count_wr(),
            pmm_reads   = pmm_read_cmd(),
            pmm_writes  = pmm_write_cmd(),
        )),
        # Queue Depths
        () -> uncore_events((
            unc_clocks  = unc_clocks(),
            pmm_rq      = pmm_rq(),
            pmm_wq      = pmm_wq(),
            dram_wq     = dram_wq(),
        )),
    ]

    params = KernelParams(
        array_size,
        1.0,
        "1lm",
        5,
        measurements,
        [16],           # only use AvX 512
        [true, false],  # Use nontemporal and standard loads and stores
        "pmm",
    )

    # Run these experiments with a PersistentArray instead of a normal array.
    runkernels(
        params,
        sz -> PersistentArray{Float32}(undef, sz),
    )
end

function dram_direct_test(;test = false)
    @assert Threads.nthreads() == 24

    measurements = [
        # Bandwidth statistics
        () -> uncore_events((
            dram_reads  = cas_count_rd(),
            dram_writes = cas_count_wr(),
            pmm_reads   = pmm_read_cmd(),
            pmm_writes  = pmm_write_cmd(),
        )),
        # Queue Depths
        () -> uncore_events((
            unc_clocks  = unc_clocks(),
            pmm_rq      = pmm_rq(),
            dram_wq     = dram_wq(),
            dram_rq     = dram_rq(),
        )),
    ]

    # Set to a small size if we're doing a debug run.
    _size = test ? 20 : 29

    params = KernelParams(
        _size,
        1.0,
        "1lm",
        10,
        measurements,
        [16],           # only use AvX 512
        [true, false],  # Use nontemporal and standard loads and stores
        "dram",
    )

    # Run these experiments with a PersistentArray instead of a normal array.
    runkernels(
        params,
        sz -> hugepage_mmap(Float32, sz, Pagesize1G()),
    )
end

