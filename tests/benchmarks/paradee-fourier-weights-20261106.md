# Paradee Fourier weight snapshots — 2026-11-06

The bounded Fourier weight snapshots are retained. All eight INT8 pairs favor the candidate: short INT8 improves 13.2% and medium INT8 6.5%, with both upper 95% endpoints below 1. FP32 control intervals include no change and both upper endpoints stay below a 5% slowdown. The bounded 3.1 MB retained payload and guarded original-primitive fallback justify retaining this change.

The INT8 export builds four Fourier matrices from small live tensors on every
execution. This change recognizes only their exact eleven-node recipe and
prepares private matrices using the original primitives. Every execution checks
all seven live operand types, shapes, layouts and value bits before copying a
prepared matrix into independently owned output storage. Input mutation, input
overrides, exceptional values, unsupported modes and refused recipes keep the
original computation. Prepared matrices and operand snapshots share the existing
combined prepared-weight budget and execution contexts share only private
readonly preparation. The FP32 export has no matching generated matrices.

INT8 retains 3,102,144 additional payload bytes and copies 3,072,000 prepared
output bytes per execution. The normal copy counters and profiler report those
copies. FP32 residency and copies, and both exports' Fourier scratch requests,
remain unchanged. **These source ratios do not establish Microsoft ORT parity.**

## Complete source comparison

The prospectively declared cohort retains all 32 fresh processes: four balanced
independent control/candidate pairs for each original FP32/INT8 and 14-/62-token
workload. Every process conditions for at least 32 calls **and 40 seconds**, then
measures 33 complete Reset/Execute/waveform-copy calls. Process values are
arithmetic means; ratios are geometric paired means with two-sided 95%
log-Student-t intervals (df 3, critical value 3.182446305284263).

| Original export | Tokens | Control ms | Prepared weights ms | Candidate/control | Two-sided 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| FP32 | 14 | 25.403 | 25.489 | 1.003366 | [0.962452, 1.046020] |
| FP32 | 62 | 69.644 | 68.358 | 0.981547 | [0.952773, 1.011190] |
| INT8 | 14 | 34.044 | 29.551 | 0.868023 | [0.838816, 0.898247] |
| INT8 | 62 | 74.146 | 69.353 | 0.935358 | [0.877126, 0.997456] |

Voice is 0; speed is 1.0 for 14 tokens and 0.8 for 62 tokens. No pair was
excluded, added or replayed after its score was observed. The original exports
retain stochastic noise. Accuracy is qualified separately with captured-noise
original-checkpoint and waveform-only companions.

## Qualification and identities

Control source is `8f457501`. The candidate adds the thirteen-file patch SHA256
`470de5d4a23d20e947f03ab366eba68cb46ac67d97a11675f31e55d65e737854`.
Both use the same GraphBench caller, original models, inputs and output-copy
boundary on CPU 2 of `onnx-bench`, Standard_D4as_v7, AMD EPYC 9V45, under default
.NET 11 RC3 options. No numerical runtime override flags are used for timing.

Both exact Linux framework products pass all six private model facts over all
eight inputs, 54 Fourier tests, 24 harmonic tests and 28 independently checked
native forward-operator regions. Candidate products also pass all 21 new
recipe tests. All 48 original/checkpoint/waveform-only path conditions confirm
the intended prepared matrices, exact retained/copied payload deltas and
unchanged scratch histograms. Local development gates pass 6,240 ordinary
tests, six private facts, 120 focused hardware-disabled tests and 343 native
differential cases per framework. Explicit model/hardware skips remain recorded.
These focused hardware-disabled tests do not replace the final full model
matrices on exact committed products.

The [latest native comparison](paradee-runtime-20261106.md) belongs to an earlier
runtime. A fresh final native cohort and the remaining exact-product,
established-model, full-routine and green-CI controls remain required. Numerical
bounds are unchanged.

The [complete JSON](paradee-fourier-weights-20261106.json) retains every worker
clock, output identity, manifest, source/product hash and qualification receipt.
Clocks use the declared four-column encoding. The independently audited VM
release records verified complete archives and an empty worker set.
