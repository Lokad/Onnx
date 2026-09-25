# Isolate the packed-weight decision from the shared batched dispatcher

The candidate inserts TryRunOwnedPackedBatches at the start of the shared
RunBatchedFloatMatMul method. In both final candidate listings, the helper
has been inlined: its shape/option tests and prepared-weight branch appear
inside the dispatcher, including calls to OwnedPackedTensor.Resolve and
RunOwnedPackedRows. Both release listings lack that route.

| Process | Product | Initial batched bytes | Intermediate batched bytes | Final batched bytes | Final entry stack subtraction |
|---|---|---:|---:|---:|---:|
| a | current | 3552 | 3479 | 10422 | 664 |
| b | candidate | 3695 | 3622 | 11777 | 984 |
| c | candidate | 3695 | 3622 | 11778 | 984 |
| d | current | 3552 | 3479 | 10170 | 664 |

The candidate adds exactly 143 bytes to each initial/intermediate batched
version. The final candidate bodies are about 1.4 KB larger in this capture
and reserve more stack. This is a concrete product difference. It is not
proof that this difference caused the older 7.77% regression.

Fresh same-product processes also differ independently. Release A inlines
GCHandle.Free internals where release D calls GCHandle.Free; candidate B/C
retain the same call targets apart from register selection. All final target
listings describe Synthesized PGO. The matrix dispatcher has the same
source in both products; its native sizes still vary. Avoid interpreting
all code-size variation as one product change.

One bounded intervention follows: remove the owned-packed decision from
RunBatchedFloatMatMul and place the same decision immediately before each
of its two existing calls. Keep all arguments, destination clearing, helper
bodies, arithmetic, model preparation and implementation flags unchanged.
This should restore the shared method's original compiled body while
retaining the successful Parakeet route. Prove that structural prediction
first, then run the original correctness and performance gates once on the
new candidate. Reject it if those gates fail; do not search adjacent variants.

The listing reader needed one additive naming correction: the first batched
body prints Instrumented Tier0 but its CLR load reports QuickJitted. Both
labels, profile-counter calls and OSR patchpoints are preserved, and all
same-process identity/order/size checks remain exact. Microsoft describes
initial instrumentation for loop methods in its
[instrumented-tier design](https://github.com/dotnet/runtime/blob/main/docs/design/features/DynamicPgo-InstrumentedTiers.md).

Byte-output limit: the four final batched listings omit 22–26 bytes relative
to their printed group offsets and native-size totals. The printed bytes
are therefore not a complete memory image. Preserve those gaps; do not
invent padding or claim byte-for-byte reconstruction. Complete textual
listings and exact CLR code-size associations remain available. No further
capture is needed to establish the extra owned route and frame growth.

[All listings and clocks](report-20260925.md), [inspection evidence](inspection-20260925.json).

No score, release admission, runtime optimization setting or warmup policy changes.
