# Paradee pooled Gather experiment — 2026-11-06

This experiment is rejected and remains unmerged. It routes float Gather outputs
through the existing graph pool, retaining public APIs, numerical behavior,
independent ownership and the existing cache capacity. Other element types and
pool-disabled execution keep their allocation paths.

Short INT8 improves by 4.7%, but short FP32 slows by 2.3%. Both confidence
intervals exclude no change: all six short INT8 pairs favor the proposal and
all six short FP32 pairs favor the control. Both medium intervals include no
change. Allocation falls by about 3.16 MB per call in both short cases; that
does not establish a common latency benefit. The default runtime stays unchanged.

## Complete source comparison

All 56 prospectively fixed workers remain: eight conformance workers and
48 timing workers, six balanced independent process pairs per original
FP32/INT8 and 14/62-token case. Each timing process conditions for at least
32 calls **and 40 seconds**, then measures 33 complete Reset/Execute/output-copy
calls. No observation was removed, added or replayed. Voice is 0; speeds are
1.0 and 0.8 for the two lengths.

The existing Standard_D4as_v7 AMD EPYC 9V45 VM runs one numeric worker on CPU 2
with default .NET 11 RC3. Both variants use the identical GraphBench caller
and dependency closure. Times are arithmetic means of six process means;
ratios are geometric paired means with two-sided 95% log-Student-t intervals,
df 5 and critical value 2.570581835636305.

| Original export | Tokens | Control ms | Candidate ms | Candidate/control | Two-sided 95% interval | Favorable pairs |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| FP32 | 14 | 22.895 | 23.428 | 1.023139 | [1.001423, 1.045326] | 0/6 |
| FP32 | 62 | 60.889 | 61.490 | 1.009718 | [0.980008, 1.040329] | 1/6 |
| INT8 | 14 | 26.908 | 25.632 | 0.952572 | [0.926006, 0.979901] | 6/6 |
| INT8 | 62 | 62.773 | 61.998 | 0.987740 | [0.961294, 1.014913] | 4/6 |

This is the first pool comparison over fdfe, independently of the selected
packed-tail change in 6d7. No source ratio is applied to a Microsoft ORT score.

## Qualification and identity

Base is `fdfe3137520f609162181f9d94b9f1fcdf61b4ef`. Only the
provider, tensor Gather helper, Node dispatch and one regression-test file
change. All 143 canonical baseline runtime/global bodies are verified.
The reused control Core is `093b71244c70e84dae7ebdcd8fcbe35e831d9d833c10c5be67af0b0de65ab89b`,
with its original bb23-plus-071-patch compiler identity retained. Candidate
Core is `643bd06654152a289085f9d5eeae14f478f11917b3379a9e7b8024014dd5fbaf`;
common caller is `6ba70c4c2ea082804bc227fdb4caeeba6ae77a54db7364aff19d91a2e4ea7297`.
The JSON records all source, product, model and input hashes.

Both actual Linux framework products pass 317 default facts without skips,
287 focused scalar facts with 24 hardware skips, 48 existing path conditions,
14 harmonic self-checks, 16 captured harmonic operands and 28 independently
checked native forward regions. All seven Gather facts pass, including dirty
buffers, layouts, scalar/multidimensional and Int32/Int64 indexes, live inputs,
held outputs, reset/failure, pool-disabled execution and empty geometry.

Each exact local product passes 6,253 ordinary tests with the same 73 explicit
skips, 312 default tests with five AVX-512 skips, 287 focused scalar tests with
24 hardware skips, all six full hardware-disabled model facts over all eight
inputs twice, and 343 native differential cases. Numerical limits remain
operator scaled error at most 1e-4, waveform peak error at most 1e-3 and SNR
at least 50 dB. Actual AVX-512 worst peak is 0.0005531 and minimum SNR 61.09 dB.

## Separate diagnostics

Two fixed short profiles retain five unprofiled and five profiled observations,
plus the explicit profiler-path warmup and all earlier conditioning calls.
Unprofiled execute allocations are 21,133,104 bytes for FP32 and 24,348,648 bytes
for INT8, respectively 3,161,152 and 3,170,312 fewer bytes than the separate
074 diagnostic on the same baseline. These phase counts diagnose allocation;
latency conclusions use only the complete paired source cohort above.

Two fixed ten-second late traces contain actual float GatherValues frames in
39 of 6,963 FP32 and 36 of 6,986 INT8 main-thread samples, with zero lost events.
The unchanged default caller reports positive pool reuse throughout, and the
audited Node dispatch passes its active pool. These prove original-graph path
admission, not exact CPU utilization or a speed gain. All trace clocks remain
separate from source scores.

## Evidence and remaining work

The [complete JSON](paradee-gather-pool-20261106.json) retains all clocks,
supervision, process pairs, qualification, diagnostics, source identities and
actual ownership/release proofs. Full source, products and raw traces remain
in the SHA-bound 169,715,261-byte archive, SHA256
`ca062d57455d8382773386ca8fcaeb043e7ceb39b1a46787ba2db20e373377b5`.
Actual VM release was 19:20:38.795887 UTC after full local SHA/size verification
and a global empty-worker audit. Only the fully exported own workspace was deleted.

An independent post-export checker initially missed the caller's extra
profiler-path warmup. The original failure is preserved; only that unfinished
read-only audit was corrected, without repeating any measurements.

The [latest actual native checkpoint](paradee-runtime-20261106-e42.md) retains
its original runtime. Both 62-token gates pass and both 14-token gates fail.
Fresh final-product ORT comparisons, established-model controls and the
complete seven-model routine below 60 minutes remain required. Reduced
allocation and an INT8-only gain do not close the all-four parity objective.
