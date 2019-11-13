# Collection of the actual benchmarks that are run.

"""
Run the set of benchmarks for sequential and random array access for arrays that fit within
the DRAM cache in 2LM Mode.
"""
function run_2lm_fit_in_dram()
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
    )

    runkernels(params; delete_old = true)
end

"""
Run the set of benchmarks for sequential and random array access for arrays that do not fit
within the DRAM cache in 2LM Mode.
"""
function run_2lm_exceeds_dram()
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
        32,     # Around 17G per threads (2^34 * 4 B). Total array size ~412 GB.
        1.0,    # Sample time: 1 second
        "2lm",  # Run this in 2LM
        2,      # 2 iterations - these take quite a while.
        measurements,
    )

    runkernels(params; delete_old = true)
end

function pmm_direct_test()
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
        29,     # Back to small arrays :D
        1.0,
        "1lm",
        5,
        measurements,
    )

    # Run these experiments with a PersistentArray instead of a normal array.
    runkernels(params, PersistentArray{Float32}; delete_old = true)
end
