# Correctness and maintenance qualification, October 20, 2026

The starting revision is `aaf4f04a`. Changes are committed as `ac0d4660`
(tensor views), `f19868a5` (imports and BERT loading), `0b5e90a7` (optimizer and
joins), `e93de972` (ResNet reference), and `e8b37f35` (benchmark compatibility).

## Correctness

Destination overlap checks now compose every slice/broadcast parent mapping.
Previously, multiplying a nested 2×2 slice by identity into overlapping storage
was accepted and zeroed both input and output. Rejection now occurs before any
write. Unknown nonempty layouts are treated conservatively; known disjoint
views remain usable. Reversed-stride slice decomposition also visits axes in
the correct order: an explicit regression previously produced `[3,4,3,4]`
instead of `[3,5,3,5]` through a broadcast of a column-major slice.

`OnnxImport.LoadOrThrow` has file and packed-budget overloads. Each failure owns
its message and original inner exception. Data constructors use this API;
nullable `Load` and its documented process-global compatibility diagnostics
remain available. The API comparison preserves all 2,830 existing public and
protected declarations and common implementation flags, with only these two
intentional additive methods. Concurrent error tests verify caller filenames
and original causes after other loads complete.

## Simplification

The thirteen optimizer passes form one immutable ordered table. Registration
flags, mutable registry/disable state and per-pass locks are removed. Pass names,
order, notes, eight-round cap, fact rebuilding and per-run/environment disable
controls are preserved. Sequential/concurrent loads and fusion tests pass.

Named BERT tokenizers share a lazy load per key. Cache lookup is brief; network
work and waiting occur outside the global gate. Failed entries can retry, with
identity-checked removal protecting replacements. Offline tests verify cached
and unrelated keys remain available during a blocked load, same-key work runs
once, and failures retry. Encoding retains its per-tokenizer synchronization.
Join helpers use `string.Join` and traverse single-use iterators once.

## ResNet numerical contract

The previous hash was introduced in `4815a2ba` as a snapshot of managed output
behind an approximately matching ORT mean. Its FMA guard did not bind the hash
to a kernel implementation. The current automatic, scalar/SIMD and native ORT
paths produce different stable hashes. Disabling tiering does not change the
current automatic result, so transient tiering does not explain this mismatch.

Full independent ORT 1.29.0 outputs for constant and ramp inputs now cover every
one of 2,048 components per case. Native generation is sequential, CPU-only,
one thread, optimizations enabled, model-hash checked and repeated three times.
All six managed case/mode combinations are close: maximum observed absolute
error is 4.18e-6. The test uses `1e-5 * max(1, abs(reference))` per component,
retains original mean/spot checks, and separately verifies bit-identical managed
repeats and input/held-output ownership. Production ResNet arithmetic is unchanged.

A 0.001 perturbation at unsampled component 1234 shifts the mean by only 4.88e-7,
passing its former 1e-6 bound, but fails the complete vector comparison. The
expected vectors come solely from native ORT; no current managed hash is
substituted. See [the reference producer and contract](../tests/resnet/README.md).

## Qualification

Final normal tests pass 403 tensor and 3,621 Backend cases; scalar tests pass
395 tensor and 3,424 Backend cases. Both gates have zero failures. The full
developer solution compiles. Python maintenance fixtures pass all 198 cases.
Focused import/optimizer/BERT tests pass 139 cases; view checks pass 106. Native
differential conformance passes 309 cases and e5 passes nine. Package contents
and the independent package consumer pass; Data remains nonpackable.

The Gemma benchmark builder also freezes linked Data sources to the control
revision. Current Data may use APIs absent in an older control core; both
measured engines now share the same compatible compiled consumer. E5 candidate
`e93de972` and Gemma candidate `e8b37f35` have identical Core source trees.

The [exact source-tip CI](https://github.com/Lokad/Onnx/actions/runs/39457364919)
passes all nine jobs without retries. All 225 complete Gemma arrays match the
pinned control byte-for-byte across
scalar/SIMD/default, and all six runs pass frozen native references, including
mixed batches and the 512-token boundary.

## Practical CPU comparison

Two balanced process pairs run serially on the existing AMD VM, with one compute
worker at a time. Gemma uses ORT 1.29.0 and e5 its existing ORT 1.23.2 runner.

| Case | Control | Candidate | ORT | Paired candidate/control |
|---|---:|---:|---:|---:|
| Gemma FP32, 128 tokens | 166.04 ms | 165.67 ms | 167.73 ms | 0.9978 |
| Gemma FP32, 512 tokens | 787.26 ms | 786.52 ms | 735.92 ms | 0.9991 |
| e5, 30 tokens | 10.02 ms | 10.14 ms | 7.66 ms | 1.0117 |

Gemma's median differences are below 0.3%. E5's two pair ratios are 1.0090 and
1.0143: a small increase retained explicitly in this short engineering check.
No throughput gain is claimed and published benchmark headline figures stay
unchanged. The correctness fixes are accepted with this minor timing variation.

Independent reload validates all 24 workers and 792 timed calls, including file
bindings, pinned sources/cores, shared runner/native hashes, numerical checks,
raw samples, affinity and process accounting. Maximum foreign CPU fraction is
0.0039282 (0.393%). The controller and all compute workers are terminal; the VM
is idle. The 260-file evidence archive has SHA-256
`6714ca8fe934de986f0f794acc42ec5dee2adea05d3f0b8d68384bcbe518e2f3`.
Local evidence is under ignored `artifacts/engineering-fixes-20261020`.
