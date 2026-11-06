# Paradee packed matrix tails — 2026-11-06

Eligible compact matrix tails process sixteen columns for eight rows with
AVX-512, sharing the live input broadcasts used by two old eight-column
passes. Reduction order and FMA operands are preserved. Remaining rows,
eight-column lanes and masked separate multiply/add arithmetic retain their
existing paths. Packing, scratch, retained storage, options and APIs are unchanged.

The change is retained. FP32 at 62 tokens has about 2% lower latency, with
its 95% interval wholly below 1. The other three estimates are 0.2–1.6% lower,
but their intervals include no change. Those gains remain inconclusive.
All 24 process pairs remain; 17 favor the candidate. No material regression
is observed. Fresh Microsoft ORT parity remains unproved.

## Complete source comparison

All 56 prospectively fixed workers are retained: eight conformance workers
and 48 timing workers, six balanced independent pairs per original FP32/INT8
and 14/62-token case. Each timing process warms for at least 32 calls **and
40 seconds**, then measures 33 complete Reset/Execute/output-copy calls.
No observation was dropped, added or replayed. The existing Standard_D4as_v7
AMD EPYC 9V45 VM runs one numeric process on CPU 2 with default .NET 11 RC3
and the identical GraphBench caller and dependency closure. Voice is 0;
speeds are 1.0 and 0.8 for the two lengths.

Times are arithmetic means of six process means. Ratios are geometric paired
means; two-sided 95% log-Student-t intervals use df 5 and critical value
2.570581835636305.

| Original export | Tokens | Control ms | Candidate ms | Candidate/control | Two-sided 95% interval | Favorable pairs |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| FP32 | 14 | 22.993 | 22.787 | 0.991040 | [0.965075, 1.017704] | 4/6 |
| FP32 | 62 | 61.380 | 60.189 | 0.980528 | [0.963967, 0.997373] | 5/6 |
| INT8 | 14 | 26.978 | 26.554 | 0.984447 | [0.951726, 1.018293] | 4/6 |
| INT8 | 62 | 62.930 | 62.812 | 0.998153 | [0.981315, 1.015281] | 4/6 |

One interval lies wholly below 1; three include no change. Every upper
endpoint is below 1.019. These are source comparisons and establish no new
Microsoft ORT ratio.

## Qualification and identities

Source base is `fdfe3137520f609162181f9d94b9f1fcdf61b4ef`, with only
MathOps.PackedColumnTails.cs and PackedColumnRemainderTests.cs changed.
Patch SHA256 is `0fce43fe2e876b08e008c51b94b53cebfc015b6cda6165d869de0447aa7ee697`.
Control Core `093b71244c70e84dae7ebdcd8fcbe35e831d9d833c10c5be67af0b0de65ab89b`
retains its original bb23-plus-071-patch compiler identity. All 143 canonical
runtime/global bodies match fdfe; it is not relabeled as a newly compiled commit.
Timed candidate Core is `cd5cd5b3d55519bc4258c968cb4be5449cd2df7980338db95568f7385b703a5d`;
common caller is `6ba70c4c2ea082804bc227fdb4caeeba6ae77a54db7364aff19d91a2e4ea7297`.
The JSON records both frameworks, all sources and model/input hashes.

Both actual Linux products pass 311 default tests without skips, including
all six private model facts over all eight input conditions twice. Each passes
280 focused scalar tests with 25 explicit hardware skips, 48 path/cache/copy/
scratch/retained-weight conditions, 14 harmonic self-checks, 16 captured harmonic
operands and 28 independent native Fourier regions.

The independent packed-tail oracle checks 1,995,315 raw float patterns per
framework with live edits, offsets, guards and row/panel boundaries. Finite,
signed-zero and infinity bits remain exact. NaN classification is preserved;
534 NaN payload differences versus the scalar oracle are counted per framework.

Each exact local product passes 6,247 ordinary tests with the same 73 explicit
skips, 306 default tests with five AVX-512 skips, 280 focused scalar tests with
25 hardware skips, all six full hardware-disabled model facts over all eight
inputs twice, and 343 native differential cases. Operator scaled error remains
at most 1e-4, waveform peak error at most 1e-3 and SNR at least 50 dB. On actual
AVX-512 the worst controlled waveform peak is 0.0005531 and minimum SNR 61.09 dB.

Two fixed post-cohort original-short traces contain the new helper in the
declared ten-second late execution intervals: 135 of 7,081 main-thread samples
for FP32 and 138 of 7,070 for INT8, with zero lost events. These prove graph
admission, not exact CPU utilization or speed. Trace clocks are separate from
the source comparison.

## Evidence and remaining work

The [complete JSON](paradee-packed-tails-20261106.json) retains every worker
clock and supervision record, all paired process means, identities,
qualification and actual ownership/release proofs. Full source, products and
raw traces are retained in its SHA-bound 169,372,020-byte evidence archive,
SHA256 `8274ce528d95e1ce86234dcf6555deb0b7c8469f2f8c895701cb7f51bb489029`.
Actual VM release was 18:14:09.805664 UTC after complete local hash/size
verification and a global empty-worker audit. Only the fully exported own
workspace was deleted; the other project's assets were preserved.

The [latest actual native checkpoint](paradee-runtime-20261106-e42.md) still
belongs to e42: both 62-token gates pass and both 14-token gates fail. Never
multiply source ratios into those scores. A fresh committed all-four ORT
comparison with point estimates and upper 95% endpoints below 1.03, final
product numerical qualification, established-model controls and the complete
seven-model routine below 60 minutes remain required.
