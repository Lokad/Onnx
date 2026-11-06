# Paradee vector Exp and squaring — 2026-11-06

The proposal reuses the existing vector elementwise operations for standard
dense float Exp and Pow with a live rank-zero float exponent of 2. Other
types, layouts, exponents and execution modes retain their existing paths.
The default float results may differ in rounding; all existing operator and
complete-waveform numerical bounds remain unchanged. No cache, retained
weights, allocator, package or public option is added.

The change is retained. Both 62-token workloads have 5.7–5.9% lower latency,
with 95% confidence intervals wholly below 1. All 24 process pairs are
retained; 21 favor the proposal.

The 14-token estimates are 2.1–2.9% lower, but both intervals include no
change. Short FP32 permits up to 4.26% slowdown. These short-input gains
remain statistically inconclusive.

The small implementation uses existing operations and passes the numerical
and ownership gates on both frameworks, including the actual AVX-512 path.
No repeatable material regression is observed. Fresh ORT parity remains
unproven.

## Complete source comparison

All 56 prospectively fixed workers are retained: eight conformance workers
and 48 timing workers, six balanced independent process pairs per original
FP32/INT8 and 14/62-token case. Every timing process warms for at least
32 calls **and 40 seconds**, then measures 33 complete Reset/Execute/output-copy
calls. No pair is dropped, added or replayed. The existing Standard_D4as_v7
AMD EPYC 9V45 VM runs one numeric worker on CPU 2, using default .NET 11 RC3
settings and the identical GraphBench caller and dependency closure.

Times are arithmetic means of six process means. Ratios are geometric paired
means; two-sided 95% log-Student-t intervals use df 5 and critical value
2.570581835636305. Voice is 0; speeds are 1.0 and 0.8 for the two lengths.

| Original export | Tokens | Control ms | Candidate ms | Candidate/control | Two-sided 95% interval | Favorable pairs |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| FP32 | 14 | 24.814 | 24.304 | 0.978967 | [0.919216, 1.042601] | 4/6 |
| FP32 | 62 | 65.989 | 62.236 | 0.943154 | [0.917853, 0.969152] | 6/6 |
| INT8 | 14 | 28.415 | 27.573 | 0.970641 | [0.929550, 1.013548] | 5/6 |
| INT8 | 62 | 67.817 | 63.810 | 0.940908 | [0.905443, 0.977762] | 6/6 |

2 of four intervals lie wholly below 1; 2 include
no change. These source comparisons do not establish new Microsoft ORT ratios.

## Qualification and identities

Source base is `bb23fbc70e38817b8e083032643aa56e13fecc36`, with only the two
qualified provider functions and two regression-test files changed. Canonical
patch SHA256 is `ebcc62534c2163cfea451a9d6a84900e5babd8ac42080f89b73f36d4dccd2d37`.
The control truthfully reuses the previously qualified live-panel product,
Core `c9ba46e62b232417c48e4c104acc2d33a9a2c917f4eb8c873a064bec31cead05`. Its original e779-plus-patch
compiler identity is preserved; all 143 canonical runtime/global source bodies
are checked against bb23 before reuse. It is not relabeled as a newly compiled
commit. Timed candidate Core is `093b71244c70e84dae7ebdcd8fcbe35e831d9d833c10c5be67af0b0de65ab89b`;
the common caller is `6ba70c4c2ea082804bc227fdb4caeeba6ae77a54db7364aff19d91a2e4ea7297`.

Both actual Linux framework products pass 308 default tests without skips,
including all six private model tests over all eight inputs twice. Focused
hardware-disabled qualification has 280 passes and 22 explicit hardware skips.
Actual AVX-512 coverage includes the five tests unavailable on the local host,
48 full path/cache/copy/retained-memory/scratch conditions, 14 harmonic
self-checks, 16 captured harmonic operands and 28 independently checked native
Fourier regions per framework. The intended optimized paths are exercised.

Each exact local framework product passes 6,246 ordinary tests with the same
73 explicit skips, 303 default tests with five explicit AVX-512 skips, 280
focused scalar tests, all six full hardware-disabled model tests over all
eight inputs twice, and 343 native differential cases. Operator scaled error
stays at or below 1e-4; complete controlled waveforms keep peak error at or
below 1e-3 and SNR at or above 50 dB. Input, held-output, reset, live exponent,
rank/layout, exceptional-value and fallback checks remain in the qualification.

## Evidence and remaining work

The [complete JSON](paradee-arithmetic-20261106.json) retains all worker clocks,
supervision records, paired process means, source/product/manifest identities,
qualification hashes and actual VM binding/release proof. Full source and
qualification evidence is in the SHA-bound archive recorded there. A local
patch transport mistake and a preclock scalar-filter count-check mistake were
preserved and corrected without replaying passed qualification or any timing.
The fully verified preparation was released before a fresh lease ran this
first timing cohort from exactly the saved binaries.

Separate post-cohort ORT operator traces use eight warmup and five diagnostic
calls per original short export. Every trace and all 13 observations remain
in the complete archive; their aggregate table is diagnostic only and is not
mixed into any benchmark score.

The [latest actual native checkpoint](paradee-runtime-20261106-e42.md) still
belongs to e42: both 62-token gates pass and both 14-token gates fail. Source
ratios must not be multiplied into those scores. A fresh committed all-four
ORT cohort with point estimates and upper 95% endpoints below 1.03, exact
final-product qualification, established-model controls and the full
seven-model routine below 60 minutes remain required.
