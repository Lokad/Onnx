# Paradee live LSTM panel packing — 2026-11-06

The change reuses the existing 8x8 SIMD transpose to build live LSTM weight
panels. It preserves every float bit, per-invocation reads, panel layout,
projection arithmetic, pooled scratch and output ownership. AVX-unavailable
execution retains the scalar packer. It adds no retained weights or cache.

The change is retained: all four latency estimates are lower and 19 of 24
process pairs favor it. The local packing-only diagnostic takes about one
quarter of the preceding packer's time on the actual H96 weight dimensions.
**Every whole-model confidence interval includes no change.** The end-to-end
gain is statistically inconclusive; the largest compatible upper slowdown is
2.72%. The small copy-only implementation and its independent bit/ownership
qualification justify keeping it. Fresh native ORT parity remains unproven.

## Complete source comparison

The prospectively frozen cohort contains all 48 timing and eight conformance
workers: six balanced independent control/candidate pairs for each original
FP32/INT8 export and 14/62-token workload. Each timing process warms for at
least 32 calls **and 40 seconds**, then measures 33 complete
Reset/Execute/waveform-copy calls. No pair was excluded, added or replayed.

Displayed times are arithmetic means of the six process means. Ratios are
geometric paired means; two-sided 95% log-Student-t intervals use df 5 and
critical value 2.570581835636305.

| Original export | Tokens | Control ms | SIMD packing ms | Candidate/control | Two-sided 95% interval | Favorable pairs |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| FP32 | 14 | 25.922 | 24.736 | 0.954221 | [0.910072, 1.000513] | 5/6 |
| FP32 | 62 | 68.840 | 68.142 | 0.990059 | [0.964097, 1.016721] | 4/6 |
| INT8 | 14 | 29.745 | 28.873 | 0.969788 | [0.915607, 1.027176] | 5/6 |
| INT8 | 62 | 68.011 | 66.880 | 0.983290 | [0.961684, 1.005381] | 5/6 |

Voice is 0; speeds are 1.0 and 0.8 for the two lengths. Both products use
the same GraphBench caller and dependency closure, one logical CPU (CPU 2)
on the existing Standard_D4as_v7 AMD EPYC 9V45 VM, and default .NET 11 RC3
options. The original stochastic exports and complete-copy boundary stay
unchanged. Accuracy is checked separately using the frozen controlled-noise
original-checkpoint and waveform-only companions.

## Qualification and identities

The control is the exact qualified Linux `e42a9028` product used by the
[latest native checkpoint](paradee-runtime-20261106-e42.md), Core SHA256
`e914a09606671c1c17b390ac0ecf7b52fa8c5cea51f3940cd203b56ad62bb7f7`.
Candidate source is the minimum two-file runtime/test patch over report-only
`e779ea02`; runtime and GraphBench source are unchanged between e42 and e779.
Patch SHA256 is `8dde1d37b3969518faf98a8624712a5241c1880c933d94702d31a01def43d173`.
The actual timed candidate Core is
`c9ba46e62b232417c48e4c104acc2d33a9a2c917f4eb8c873a064bec31cead05`.
The identical caller for both is
`6ba70c4c2ea082804bc227fdb4caeeba6ae77a54db7364aff19d91a2e4ea7297`.

Before timing, both actual Linux framework products pass 205 default facts,
including all six full model facts over all eight inputs twice, and 199
focused hardware-disabled LSTM facts. The new packing tests check raw bits,
NaN payloads, signed zero, offsets, tile/block tails, empty dimensions, source
guards, live input updates and invalid-shape refusal on this AVX-512 host.

The exact isolated Windows products independently pass 6,242 ordinary tests
with the same 73 explicit skips as the control, all six default model facts,
all six full hardware-disabled model facts, 199 focused scalar LSTM facts and
343 native differential cases per framework. Default waveform peak error is
0.000553073 with minimum SNR 61.092 dB; full scalar peak is 0.000574065 with
minimum SNR 59.889 dB. All numerical limits are unchanged. The earlier local
restore/CLI-prerequisite failures and corrected gates remain recorded.

The separate local packing-only diagnostic uses an AVX2 Windows host and
retains all 325 raw-bit/guard shapes and 144 fixed timing observations. Its
numbers describe that inner copy routine. The table above is the complete
VM source comparison. No earlier source/native clocks are pooled into it.

## Evidence and remaining work

The [complete JSON](paradee-lstm-panels-20261106.json) retains every worker
clock, supervision record, output identity, manifest and source/product/
qualification hash. Clocks use the declared four-column encoding.
The full archive was independently SHA/size-verified before the actual
14:04:54 UTC release and empty-worker audit. Only the fully exported owned
root was removed. The original failed Windows extraction and the corrected
handling of its two protected Linux fixture links remain preserved.

The [native table](paradee-runtime-20261106-e42.md) still belongs to e42:
both 62-token gates pass and both 14-token gates fail. Source point estimates
must not be multiplied into native scores. A fresh final four-case ORT cohort,
exact final-product gates, established-model controls and the complete
seven-model routine below 60 minutes remain required.
