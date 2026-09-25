# Parakeet: integrated packed final-row contracts

**All 84 focused checks pass.** The isolated candidate computes the remaining
row directly from its existing packed weights. Both actual matrix shapes and
all seven affected sequence lengths give bit-identical results with zero
reconstruction bytes and zero weight scratch bytes in the checked graph calls.

| Actual instruction mode | Passed checks | Skipped |
|---|---:|---:|
| Normal | 41 | 0 |
| AVX512 disabled | 41 | 0 |
| All hardware intrinsics disabled | 2 | 0 |

Each suite includes loaded-product hashes, runtime, CPU affinity and instruction
availability checks. Existing ownership, logical tensor, mutation, alias,
collection and fallback contracts remain covered. The new helper is not called
when hardware intrinsics are unavailable.

The compiled audit compares the helper's full normalized IL and implementation
flags with the successful standalone proof. They match exactly. It permits
changes to the private row loop and its two callers, removal of the dense
reconstruction method, and addition of the helper. All other 3,273 Core methods
and all 697 Data methods remain unchanged. The audit explicitly accounts for
41 compiler-generated name shifts while preserving instructions, literals,
branches, stack declarations and flags. Public interfaces remain unchanged.
The build retains two established nullable warnings and introduces none.

This implements one difference identified against the installed ORT source:
ORT's row loop continues consuming prepared weights; the previous Lokad candidate
reconstructed a full 16 MiB weight matrix to handle one row. Actual earlier
counters found 609 such reconstructions across the 20-clip corpus. This candidate
has passed focused contracts; its actual model traffic is not yet measured.
The earlier unstable copy timings remain unsuitable for predicting a saving.

Core identity is 49901366 and Data identity 01e9e784. Both were built in isolation
on the VM with SDK 10.0.204 and tested on runtime 10.0.8. Peak monitored test RSS
was 727,199,744 bytes. Both stage owners and all workers are terminal; the 581
build files and 598 contract files were collected and reviewed once.

Next, reuse the existing census consumer with these products to confirm the
87 owned weights and 37 retained maps, then complete native/public model checks,
confirm zero reconstruction on the actual corpus, and run the matched application
comparison against selected M73 and fresh ORT. The original performance gates
and unresolved e5 release controls remain. No new application speedup or release
admission is claimed; BENCHMARK.md remains on the qualified product.

[Exact identities and test names](contracts-20260925.json),
[standalone proof](proof-20260925.md),
[build and audit tools](../packed-final-row-build/README.md).

Closure: `6295ad30835b7a2a1694b580a8e1447a6a0cdb28828a5960fa6e0e7c3628576f`.
