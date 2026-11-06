# Paradee wider forward Fourier source comparison — 2026-11-06

The wider forward transform is retained. The FP32 14-token workload improves by about 3.2%, with its confidence interval excluding no change. The other three intervals include 1; short INT8 has a 1.1% slower point estimate. All four upper endpoints remain below a 5% slowdown. The confirmed FP32 gain and independently demonstrated component improvement justify retaining the coherent hardware path, with the remaining whole-model uncertainty explicit.

The trial processes sixteen independent frames per vector on AVX-512 hardware,
using the existing transform object and broadcasting its existing eight-lane
roots. It preserves each lane's arithmetic and retained payload. Four bounded
scratch arrays grow with vector width; their maximum admitted payload is 256 KiB.
Unsupported hardware uses the existing forward transform. Weight, mode, layout,
geometry, overlap and numerical-envelope guards remain in place. There is no
new cache, root set or public option. **These source ratios do not establish
Microsoft ORT parity.**

## Complete source comparison

The prospectively declared cohort includes all 48 fresh processes: six balanced
independent control/candidate pairs for each original FP32/INT8 and 14-/62-token
workload. Each process conditions for at least 32 calls **and 40 seconds**, then
measures 33 complete Reset/Execute/waveform-copy calls. Process times are
arithmetic means; paired ratios use geometric means and two-sided 95%
log-Student-t intervals (df 5, critical value 2.570581835636305).

| Original export | Tokens | Control ms | Wider transform ms | Candidate/control | Two-sided 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| FP32 | 14 | 26.558 | 25.707 | 0.967925 | [0.939563, 0.997143] |
| FP32 | 62 | 72.409 | 70.048 | 0.967392 | [0.897537, 1.042684] |
| INT8 | 14 | 34.587 | 34.977 | 1.011298 | [0.983830, 1.039532] |
| INT8 | 62 | 76.243 | 74.925 | 0.982717 | [0.960925, 1.005002] |

Voice is 0; speed is 1.0 for 14 tokens and 0.8 for 62 tokens. Every declared
process observation is retained; no pair was excluded, added or repeated after
inspecting its score. The original exports retain their stochastic noise.
Deterministic accuracy is established separately with the captured-noise
original-checkpoint and waveform-only companions.

## Qualification and identities

Control source is `f033c446`, runtime `90d7e145`. The candidate adds the exact
three-file patch SHA256
`7d45e2d01f4e4bec402f6d1ed15364e8e4021d02a839c6751bed31c2a69abe24`.
Both use the same GraphBench caller source, original models and inputs, complete
copy boundary, one logical CPU (CPU 2), and default .NET 11 RC3 options on
`onnx-bench`, Standard_D4as_v7, AMD EPYC 9V45. Binary hashes are recorded per
variant/framework. No runtime override flags are used for timing.

Both exact Linux framework products pass all six private model facts over all
eight inputs, including waveform-only companions. The candidate passes all 54
Fourier tests on actual AVX-512 hardware; control passes its existing 34. Both
variants pass 24 harmonic tests and separate production checks. Independently
captured operands pass all 28 native forward-operator cases per variant and
framework. All 48 model path conditions change precisely seven scratch requests
from eight to sixteen lanes, with retained payload unchanged. Local development
gates pass 6,220 ordinary tests, six private facts, 343 native differential cases,
99 focused hardware-disabled tests, 28 native forward regions and 48 model path
conditions per framework. Explicit hardware/model skips remain recorded.

The preceding independent shared-root component trial used 16 processes and
1,025 measured calls per process. Its wide/narrow ratios are about 0.777 for
FP32/62 and 0.837 for INT8/14, with both upper 95% endpoints below 0.95. Those
isolated component measurements are separate from the source scores above.
An earlier, shorter component trial had an inconclusive short INT8 interval;
its scores were not pooled into the later trial or the source cohort.

The [latest native comparison](paradee-runtime-20261106.md) belongs to an earlier
runtime. A fresh final native cohort and the remaining exact-product,
established-model and full-routine controls remain required. Numerical bounds
are unchanged.

The [complete JSON](paradee-wide-fourier-20261106.json) retains every source
worker clock, output identity, manifest, source/product hash and qualification
receipt. Clocks use the declared four-column encoding. Complete exported
evidence archives are identified by SHA256 and size; the independently audited
VM release records an empty worker set.
