# EmbeddingGemma 2 FP32 fixed-width CPU comparison, 2026-10-17

At 128 tokens the candidate takes **165.33 ms versus ORT's 168.01 ms**, with a paired latency ratio of **0.984**. The separate fresh 128-token comparison measures 165.09/168.10 ms (ratio 0.982) using the identical candidate binary.

| Input | Previous optimized baseline ms | Candidate ms | ORT ms, candidate visits | Candidate / ORT | Baseline ratio repeats | Candidate ratio repeats |
|---|---:|---:|---:|---:|---|---|
| 32 tokens | 54.58 | 54.92 | 53.62 | 1.024 | 0.998, 1.030 | 1.011, 1.037 |
| 128 tokens | 176.55 | 165.33 | 168.01 | 0.984 | 1.052, 1.054 | 0.986, 0.982 |
| 512 tokens | 786.62 | 781.72 | 737.86 | 1.059 | 1.062, 1.062 | 1.051, 1.067 |
| 32 tokens padded to 128 | 177.19 | 165.08 | 167.84 | 0.984 | 1.059, 1.058 | 0.983, 0.984 |

The earlier 1.05 ratio target remains missed for 512 tokens.
Read baseline/candidate differences alongside their repeat ranges. The 32-token case has more process-to-process variation; the new fixed-width dispatch targets 128-row projections.

All four 128-token candidate/native repeat ratios: 0.986061, 0.982049, 0.982775, 0.981387.

All four candidate visits use less time than their paired native visits. This is a small, workload-specific observed lead; four fresh candidate processes and their paired native processes describe the observed variation without establishing a confidence interval.

The 0.97 working ratio target is missed. Other token lengths are reported separately; the 128-token result does not establish a lead for every case.

## Implementation and measured sources

The candidate specializes packed FP32 projections with 128 rows and measured reduction/output widths. Constant row displacements reduce address work; the eight-row tail runs while its column panel is current. The per-output reduction and fused multiply-add order, packing format, retained weight budget, precision and final-vector behavior remain unchanged. Unsupported shapes retain the existing kernels. The measurements evaluate these changes together and do not isolate their individual contributions.

Candidate source: `98b9234cbc740bd1546754394f4a3bf51a6289ac`, Linux core SHA-256 `9b365f1f2f56b1910aff3190d751a5647251dad1c97257fa593f8d7cca1956e5`. Previous optimized baseline: `0132a460d1f1030dce53ca85d8a7c578bf8f12e5`, core `b5f0bae1562b96db91f8747b4ac839a267fba5d3df1cf56179271aeec2ab6f60`. The [earlier comparison](benchmarks-20261017.md) retains the preceding implementation and its results.

The independent 128-token screen uses immediate control `d9712d6d4478690c86f59953dbc2fe2f95d1b9af` and the same candidate as the final comparison. The candidate/native paired ratios in both comparisons provide the repeated check of the ORT result.

## Environment and timing

Existing Azure Standard_D4as_v7 onnx-bench VM, AMD EPYC 9V45, 16 GiB RAM, Ubuntu 24.04.4 x64, .NET SDK 10.0.204/runtime 10.0.8, and ORT 1.29.0 CPUExecutionProvider. Inference uses one thread on CPU2, its sibling CPU3 is idle, and supervision/services use CPUs0–1. ORT uses the optimized original graph, sequential execution, one intra/inter-op thread and spinning disabled. Retained packed-weight budget: 576 MiB.

Both engines produce an owned 768-value FP32 vector. Compute includes graph execution and the required pooling, truncation, normalization and copying. Tokenization, loading, context reset and disposal stay outside compute. The unchanged common runner warms for at least 32 calls and 30 compute seconds, then retains 33 calls in three blocks of 11.

The final comparison uses baseline/candidate/candidate/baseline revision ordering, alternating engine order, and all four cases: 32 measured workers plus two preflight workers. The independent 128-token screen adds eight measured workers and two probes. The table reports the median of process medians; ratios are medians of paired visit ratios. All 1,320 measured call times are retained across the two comparisons.

## Correctness, regressions and evidence

Every measured worker passes the original source/binary/input, pre/post vector, runtime, affinity and process-accounting validators. Managed workers load no native ORT. Separate Windows/Linux qualification covers all 15 frozen cases, four output dimensions, scalar/SIMD/default conformance and byte-identical repeats/baseline comparisons, integrated API inputs and 52 vectors, CLI, e5 numerical conformance and independent package consumption. All 423 tensor tests run on the Linux AVX-512 host. The previously reproduced Windows ResNet frozen-hash baseline failure remains recorded; Linux backend tests pass 3,736 with 44 skips. Lokad.Onnx.Data remains nonpackable.

[Exact-source CI](https://github.com/Lokad/Onnx/actions/runs/39044439155) passes all eight jobs without reruns. Raw timing warnings: 0; maximum unrelated guest CPU fraction: 0.276%. All samples, including any warnings, remain in the evidence. Both comparisons accommodate minor variation and preserve historical refused calibrations without claiming acceptance by their 3% A/A scorer.

[Machine-readable measurements](benchmarks-20261017-fixedwidth.json) include all measured workers and raw call times, repeat ranges, loading, allocations, peak memory, source/core identities and archive receipts. The common runner composite SHA-256 is `67691e5c45b9ff8f5f6e2f02564ec10092bec7d0c1cb5c099ba833a6ed0eb259`.

Complete final archive: `embeddinggemma2-fixedwide-final-v70.tar.gz`, SHA-256 `4da33d2062b8086b06f4253f81f9b434d3995b7b4c464ae765b79d8746fbb75c`. Independent 128-token archive: `embeddinggemma2-fixedwide-screen-v69.tar.gz`, SHA-256 `c66d94bdf268aae454a38787c02547a66f6ed2708f2acfdef9b4123857fd9b79`. Qualification archive: `embeddinggemma2-fixedwide-qualified-v68.tar.gz`, SHA-256 `68c30c199a5d09405c7c41b0b1db8c7ee65afa37875f5cb825ee7df352500cb0`. All archive members are verified on the VM and Windows; every measured worker median and ratio is independently recomputed. These complete archives remain under ignored artifacts/embeddinggemma2 on both machines. Temporary VM settings are restored and no owned benchmark job remains.
