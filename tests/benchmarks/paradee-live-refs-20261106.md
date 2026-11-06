# Paradee live-reference lookup experiment — 2026-11-06

Rejected; the runtime change remains unmerged.

The complete experiment suggests small possible short-input improvements (FP32 0.64%, INT8 0.83%), but all four two-sided95% source intervals include no change. Medium FP32 is effectively flat and medium INT8 has a0.46% favorable point estimate;14of24pairs favor the proposal. All correctness and late parent-path checks pass. A useful whole-model gain is not established, so reject default adoption of the lower-level dictionary-reference change, retain all observations and keep current runtime unchanged. No sample extension, pair removal, unchanged replay or newORTparity claim.

Only duplicate dictionary probes in graph live-array reference counting change.
Existing increments use CollectionsMarshal.GetValueRefOrAddDefault; decrements
use GetValueRefOrNullRef. Integer counts, zero-count key removal, sequence/view
and unknown-tensor traversal, deferred-release conditions, storage lifetimes,
arithmetic, public interfaces and dependencies remain unchanged. No dictionary
value reference is used across a structural mutation.

## Complete fixed source comparison

All 56 workers and 24 pairs remain: eight conformance workers and six balanced
independent pairs per original FP32/INT8, 14/62-token workload. Each timing
process conditions for at least 32 calls **and 40 seconds**, then measures 33
complete Reset/Execute/waveform-copy calls. No observation was dropped, added
or replayed after inspection. Numerical work runs serially on CPU 2 of the
existing Standard_D4as_v7/AMD EPYC 9V45 VM, default .NET 11 RC3, with one common
GraphBench caller/dependency closure. Voice is 0; speeds are 1.0 and 0.8.

Times are arithmetic means of process means. Ratios are geometric paired means;
two-sided 95% log-Student-t intervals use df 5 and critical value 2.570581835636305.

| Original export | Tokens | Control ms | Candidate ms | Candidate/control | Two-sided 95% interval | Favorable pairs |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| FP32 | 14 | 22.559 | 22.415 | 0.993641 | [0.981122, 1.006320] | 5/6 |
| FP32 | 62 | 59.653 | 59.688 | 1.000661 | [0.968090, 1.034326] | 3/6 |
| INT8 | 14 | 26.551 | 26.335 | 0.991702 | [0.962337, 1.021964] | 4/6 |
| INT8 | 62 | 61.646 | 61.360 | 0.995357 | [0.971362, 1.019946] | 2/6 |

These source results establish no Microsoft ORT ratio. The
[actual native checkpoint](paradee-runtime-20261106-b47.md) remains the current
comparison: both 62-token gates pass and both 14-token gates fail. Source ratios
are not multiplied into earlier native results.

## Qualification and identities

Both exact local products pass 6,247 ordinary tests with the same 73 explicit
skips, 326 default tests with five AVX-512 skips, 300 focused scalar tests with
25 hardware skips, all six FULL hardware-disabled model facts over eight inputs
twice, and 343 native differential cases. All 20 existing alias, deferred-release
and collection facts pass. Both actual Linux products pass 331 default facts
without skips, 300 focused scalar facts with 25 explicit skips, 48 path conditions,
14 harmonic self-checks, 16 captured operands and 28 independent native Fourier
regions. Numerical, duration, mutation/reset/repeat and held-output bounds are
unchanged. The fixed late trace inventory records parent-path presence only;
sampled frames do not establish CPU shares or predicted savings.

Source base is `97dad5121da71b4e8fa0bcf38901403c6a56d17a`. Only
ComputationalGraph.cs changes. The exact measured clean b47 baseline Core
`f045990b351d4685bc5bb08810e3ca415be84be2a3326385d22fb7188a15b5b5` retains its original compiler
identity; all 143 canonical runtime/global bodies match the report-only base.
Timed candidate Core is `327aa5b6e6a11579e6b2781e09141e6141b61910542c448b8ce0de2fddab0e52` and the common caller is
`0c629efcc31a6c19c44be78e538a29f4bbe9a4810b3881aa396692349f6212db`.

## Evidence

The [complete JSON](paradee-live-refs-20261106.json) retains every worker clock,
supervision record, pair, qualification, source/product identity, trace inventory
and decision. The full 169,162,641-byte archive has SHA256
`afd21f4fbcfdf635a86d7814be90dc9d09895d7d11706ccc20f3d5111d4af798`. Actual slot 052 was released at
`2026-11-06T22:38:49.716084+00:00` after full local SHA/size verification,
all 18 original model/input/reference hash checks, a global empty-worker audit
and matching-token removal. Only the fully exported own root was deleted.

Final all-four actual ORT parity, established-model controls and the actual
under-60-minute seven-model routine remain open. No package, release or tag is
produced by this experiment.
