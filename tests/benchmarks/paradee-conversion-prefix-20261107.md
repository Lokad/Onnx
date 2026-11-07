# Paradee prepared conversions — 2026-11-07

Selected for integration, subject to main/committed-product qualification and
a fresh actual ORT comparison.

Short INT8 latency is 9.96% lower, with all six process pairs favorable and a
geometric ratio of 0.900408 (two-sided 95% interval [0.853481, 0.949917]).
Short FP32 is 1.74% slower: ratio 1.017418 [1.004816, 1.030178]. This measured
cost is accepted explicitly for the short INT8 improvement; it is not called
no change. Both longer-case intervals include no change and establish no gain.
All original final ORT point and upper-interval gates below 1.03 remain required.

The runtime prepares a bounded initial prefix of initializer-only Cast,
DequantizeLinear and existing fused Fourier-weight operations through the
existing preparation cache. Original nodes and fallback positions remain.
Every source binding and node description is checked before reuse, including
checks at each original node. Unsupported types, modes, budgets or custom
reporters keep the original interpreter behavior. Public DenseTensor subclasses
are refused, preserving their conversion and failure behavior.

Prepared float values remain private. Each use has independent mutable storage;
public writes and storage exposure materialize that use's owned array. Existing
independent-output weight consumers can borrow read-only views. Harmonic and
binary arithmetic retain their dense paths through owning storage, including
built-in affine views. Names, held outputs and callback-visible writes remain.
The original INT8 optimized prefix has 378 nodes and 42,381,106 retained payload
bytes, charged against the existing combined preparation limit. This is a
retained-payload count, not a process-memory estimate. FP32 has no eligible
conversion prefix. No public setting, dependency or speech frontend is added.

## Complete fixed source comparison

All 56 workers and 24 pairs remain: eight conformance workers and six balanced
independent pairs for each original FP32/INT8, 14/62-token workload. Every timing
process conditions for at least 32 calls **and 40 seconds**, then measures 33
complete Reset/Execute/waveform-copy calls. Numerical workers run serially on
CPU 2 of the existing Standard_D4as_v7 / AMD EPYC 9V45 VM with default .NET 11
RC3 and one shared GraphBench caller/dependency closure. Voice is 0; speeds are
1.0 and 0.8. The complete source cohort took 2014.124104 seconds; this is
not the separate seven-model full routine.

Times are arithmetic means of process means. Ratios are geometric paired means;
two-sided 95% log-Student-t intervals use df 5 and critical value
2.570581835636305. No observation was discarded, replayed or added. In
particular, the unusually fast fifth long-INT8 candidate process remains.

| Original export | Tokens | Control ms | Candidate ms | Candidate/control | Two-sided 95% interval | Favorable pairs |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| FP32 | 14 | 20.546 | 20.904 | 1.017418 | [1.004816, 1.030178] | 1/6 |
| FP32 | 62 | 54.573 | 55.034 | 1.008369 | [0.978928, 1.038696] | 2/6 |
| INT8 | 14 | 24.674 | 22.218 | 0.900408 | [0.853481, 0.949917] | 6/6 |
| INT8 | 62 | 58.599 | 57.754 | 0.984742 | [0.928921, 1.043917] | 4/6 |

These are source comparisons against the retained 9a2 runtime, **not ORT
ratios**. The [actual native checkpoint](paradee-runtime-20261107-9a2.md)
remains the current ORT comparison. Source ratios are not multiplied into it.

## Qualification

Both exact local products pass 6,269 ordinary tests with the same 77 explicit
skips, 359 default facts with ten hardware skips, 333 focused scalar facts with
30 explicit skips, all six FULL hardware-disabled model facts over eight inputs
twice, and 343 native differential cases. All 21 new facts pass. They cover
ownership, storage/views, live source and between-node edits, budget/mode and
reporter refusals, original-model admission, names/errors, dense harmonic
allocation, aliases, callback mutation and initializer subclasses. An unchanged
external public probe reproduces the earlier subclass failure and passes the
retained implementation on both frameworks.

Both actual Linux products pass 368 of 369 default facts; only the
hardware-absence refusal skips. All 21 new facts, six private model facts,
333 focused scalar facts, 48 model-path conditions, 14 harmonic self-checks,
16 captured operands and 28 independent native Fourier regions pass.

Original limits remain: upstream/operator scaled error at most 1e-4, waveform
peak at most 1e-3 and signal-to-error at least 50 dB. Worst actual peak is
0.0005530733615159988 and minimum signal-to-error 61.09190685273485 dB.
FULL scalar peak is 0.0005740653723478317 and minimum signal-to-error
59.88937963037473 dB. Shape, duration, input mutation, reset/repeat/random-state
and held-output checks remain unchanged.

Both fixed late traces have zero lost events and show the declared PackedTile12
parent. The INT8 trace also shows the restored HarmonicSumCore path. They
establish no unprofiled CPU-share, all-bindings-hit or cycle-saving estimate.

## Identities and evidence

Source base is `adb39a631928051b5cc6e57065a6841f83aa691c`. The twelve source/test file
hashes, complete patch identity and original model/input hashes are in the
[complete JSON](paradee-conversion-prefix-20261107.json), together with every
worker clock, supervision record, process pair, qualification and trace proof.
The exact original clean 9a2 control Core is
`9b5b0fb7427762a69eb8c4aec01e46f4efc3db73e4f2005ab50b599161bdfb36`;
timed candidate Core is `9dc970d7c2ceec9963982ff4296cecd7f09c4132d62c82cec75e9a4217700bc5`;
common caller is `669e4ab2b56d48c3663036ae995aecbb00789b1f9277b6dff55f5b316f09fef7`.
The measured source is the exact twelve-file delta selected for integration.

The full 170,767,091-byte archive SHA256 is
`c613c21e5ec3d73ae343743008af1010199cd2fd8ac9149a8b655faf805a5f31`. Slot 063 was released at
`2026-11-07T07:36:34.119778+00:00` after full local SHA/size verification,
rehashing all 18 protected assets and checking globally empty workers. Only the
fully exported owned root was removed.

Exact committed-product qualification, fresh all-four actual ORT parity,
established-model controls, the under-60-minute seven-model routine and final
terminal-green CI remain mandatory. This source result creates no release or tag.
