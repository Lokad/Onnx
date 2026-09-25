# Parakeet owned packed weights: integration contracts

The isolated candidate passes its focused graph and numerical contracts in
normal, AVX512-disabled and hardware-intrinsics-disabled execution. It has not
run the real model or established an application speedup. The qualified release
and BENCHMARK.md remain unchanged.

The [cost diagnosis](../feed-forward-cost-results/diagnosis-20260925.md) identifies
repeated preparation of constant weights as the lead. The candidate replaces
87 currently unmapped feed-forward weights with one packed payload each, while
preserving the existing multiplication methods, 37 prepared records and 256 MiB
clone budget. Those 87 weights account for 3.043 of the measured 3.150 packing
seconds per corpus. These are diagnostic costs, not predicted application gains.

Compiled inspection permits seven existing Core edits and the transcriber
constructor edit, plus 23 new internal Core methods. Existing public declarations,
arithmetic methods and implementation flags are unchanged. The first exact-name
review failed because five inserted helper methods shifted compiler-generated
ordinals. A proved bijection accounts for all 41 renamed methods without ignoring
instruction, branch, literal-string or flag differences. Seven reviewer tests
cover that correction; the original failed review is retained.

The first native suite passed 24 of 26 cases. Two odd-row cases passed their
numerical comparisons but failed an accounting assertion: graph execution uses
its own reporters, leaving the caller's reporters unused. The correction reads
the context's LastCopyBytes and LastScratchBytes. Only four instructions in one
test method change; the other 4,671 methods in the test assembly and both product
DLLs remain unchanged.

All six affected shape cases and the runtime-identity check now pass. Together
with 19 unchanged native cases from the original run, that qualifies 26 native
cases. The previously unstarted AVX512-disabled and hardware-disabled runs pass
26 and two cases respectively, with no skips. Actual loaded assembly hashes,
CPU affinity, runtime and instruction availability are checked inside the tests.

The contracts cover replacement and invalidation, preserved prepared records,
shared contexts and held outputs, alias rejection, logical reads and writes,
fallbacks, and collection of the original array. Both actual feed-forward matrix
shapes retain bit-exact arithmetic. At 167 rows, the original odd-row remainder
uses exactly one 16 MiB dense reconstruction; 48- and 225-row cases require none.
The selected route rents no weight scratch. This still requires confirmation on
the real encoder's 87 weights and all actual sequence lengths.

Next, verify the actual graph census, existing packed payload identities,
full model and public transcript correctness, memory, and per-call reconstruction.
Only then run the matched full-application comparison. The prior e5 repeatability
failures remain unresolved and continue to prevent release promotion.

The repository inventory records 43.668 GB allocated and 53.027 GB logical file
sizes; existing NTFS compression accounts for the distinction. No new model
assets were downloaded. [Exact evidence and retained failures](contracts-20260925.json)
are produced by the [read-only publisher](publish.py). Closure: `3fc949cad141d2a1fef1e3913436920f87f3cecb92f1bce94892c1037bc080f4`.
