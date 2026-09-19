# Fourteen-row packed projection experiment — September 19, 2026

The fourteen-row candidate remains outside production. It preserves arithmetic and improves the thirty-row matrix sequence in all four runs, but the 128-row sequence regresses, individual projections are mixed, and the smaller observed gains overlap movement in the unchanged eight-row control. The predefined gate does not justify a product switch or a full-model campaign.

This artifact-only prototype compares the current twelve/eight-row composer, a generated twelve-row control with the same partition, and fourteen-row bulk tiles with existing remainder kernels. All use the same 32-column packed layout, accumulating destination and FMA order. At thirty rows, partitioning changes from twelve/twelve/three/three to fourteen/fourteen/two; the experiment does not isolate row reuse from remainder composition.

| Rows | Fourteen / original, four bank ratios | Fourteen / generated twelve, four bank ratios |
|---|---|---|
| 8 | 1.0612, 0.9930, 1.0031, 0.9892 | 1.0587, 1.0098, 1.0050, 0.9880 |
| 30 | 0.9331, 0.9766, 0.9526, 0.9688 | 0.9573, 0.9837, 0.9606, 0.9750 |
| 128 | 1.0108, 1.0048, 1.0163, 1.0109 | 1.0115, 1.0131, 1.0173, 1.0035 |
| 512 | 1.0083, 0.9984, 1.0013, 1.0000 | 1.0084, 0.9955, 0.9942, 1.0032 |

Ratios use each process's retained sample medians; below one favors fourteen rows. Eight rows use the unchanged original kernel. The bank contains 72 independently verified projection matrices and 84,934,656 packed bytes, exceeding the VM's 32-MiB L3. It includes destination clearing and computation, but not real layer activation dependencies. It is not a model benchmark.

Four fixed reversed-order fresh processes ran on CPU 2 of the AMD EPYC 9V74 VM, .NET 10.0.8, normal runtime and no forced GC. Each passed 256 finite/exceptional geometry cases and eight null-pointer refusal cases. Checks cover full output bits, scalar-FMA coordinates, nonzero and repeated accumulation, offset guards and unchanged inputs/packed weights. All 144 isolated projection records, 48 bank records and 1,344 measured sample blocks remain.

Code generation was captured in a separate process after timing. The actual target loops contain:

| Method | Code bytes | FMAs per reduction step | Broadcasts | Packed vector loads | Scalar stack loads | Vector stack accesses |
|---|---:|---:|---:|---:|---:|---:|
| PackedTile12 | 1135 | 24 | 12 | 2 | 4 | 0 |
| GeneratedTile12 | 1411 | 24 | 12 | 2 | 4 | 0 |
| GeneratedTile14 | 1677 | 28 | 14 | 2 | 6 | 0 |

There are no calls in these reduction loops. The extra scalar pointer reloads show why fitting 28 vector accumulators does not guarantee a faster complete kernel. They are a code-generation finding, not proof that they alone caused the measured regression.

The independent audit checks the complete extracted archive, source/binary identities, exact geometry/record coverage, output hashes, all samples, ordering, resources and process accounting. Maximum observed foreign CPU fraction is `0.0016346694703935165`. Supervisor 286252 and all five children are terminal; collection, numerical/timing analysis and instruction review are complete.

Product core remains the qualified c6bf781 binary, SHA256 `7653c1686419d612e2624740908a44ffdee91b4239628a979bac44f5b6b863e9`. No product source, arithmetic tolerance, default or experimental switch was added. No thirteen/fifteen/sixteen-row follow-up was selected after viewing these results.

Frozen bundle SHA256: `ea80d096f17425a094580d179d7194d6b1fd162d465d0957d0ddc77f9ae20f17`. Collected archive SHA256: `1d663e30199c67bfabe16abd8c796389f904aa5e4e5c88edbbe6c66351bfe1d0`. Independent receipt SHA256: `56e2d4d48d1d69982b0807858d5ff1a8ff4b38923dd0e60a24f88a29ceff5a30`.

All source, geometry checks, raw samples and emitted instructions remain in `artifacts/packed-fourteen-20260919`. The initial preparation refusal caused by a self-referential manifest is preserved separately; its corrected immutable payload is the one measured here. Completed writers must not be rerun.
