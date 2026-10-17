# EmbeddingGemma 2 FP32 CPU comparison, 2026-10-17

On the existing `onnx-bench` VM, the optimized implementation takes **175.8 ms versus ORT's 167.9 ms at 128 tokens**, a latency ratio of **1.047** (about 4.7% more time). These are repeated measurements of an owned 768-value FP32 embedding.

| Input | Baseline ms | Optimized ms | ORT ms, optimized visits | Optimized / ORT | Optimized ratio repeats |
|---|---:|---:|---:|---:|---|
| 32 tokens | 56.78 | 54.65 | 53.68 | 1.018 | 1.003, 1.034 |
| 128 tokens | 180.32 | 175.84 | 167.89 | 1.047 | 1.050, 1.045 |
| 512 tokens | 816.92 | 785.22 | 736.35 | 1.066 | 1.065, 1.068 |
| 32 tokens padded to 128 | 179.67 | 176.24 | 167.76 | 1.051 | 1.055, 1.046 |

The median ratio target of 1.05 is met at 32 and 128 tokens. The 512-token and padded cases remain above that target. The ratio ranges describe the observed repeats; two processes per engine/revision do not establish a confidence interval.

## Environment and measured sources

Azure `Standard_D4as_v7`, AMD EPYC 9V45, 16 GiB RAM, Ubuntu 24.04.4 LTS x64, .NET SDK 10.0.204/runtime 10.0.8, and Microsoft ONNX Runtime 1.29.0 CPUExecutionProvider. Inference is confined to CPU2 before runtime startup; its sibling CPU3 is idle, and supervision/services use CPUs0–1. ORT uses sequential execution, one intra/inter-op thread, graph optimizations enabled and spinning disabled. The packed-weight budget is 576 MiB.

Optimized source: `0132a460d1f1030dce53ca85d8a7c578bf8f12e5`, core SHA-256 `b5f0bae1562b96db91f8747b4ac839a267fba5d3df1cf56179271aeec2ab6f60`. Baseline source: `223114cfc7044ffdd7c1112672416fc949341bb5`, core SHA-256 `21e0d1d0f2d5c46cdbbd9b9b0b9c3eb6a9e2cc6db6124d11ced4bfa989829d9a`. Assets and numerical coverage are pinned in [assets.json](assets.json) and [the model lane](README.md).

## Timing and repetition

Policy `embeddinggemma2-practical-abba-v1` runs baseline, optimized, optimized, baseline, with four input cases and both engines per visit: 32 sequential measured workers plus two preflight workers. Engine order alternates between visits. Each fresh process warms for at least 32 calls and 30 compute seconds, then retains 33 timed calls in three blocks of 11. There are no concurrent builds, profilers, or observer loops.

The unchanged common runner produces an owned final 768-value embedding. Managed compute includes prepared-graph execution, masked mean pooling, truncation, unit normalization and copying. ORT uses the optimized original graph with matching final-vector ownership. Loading, tokenization, context reset and disposal are outside compute timing. The table takes the median of the two process medians; ratios are medians of each visit's managed/native ratio.

All pre/post vector checks and exact input checks pass through the original worker validators. Managed workers load no native ORT. Every raw sample is retained, and no raw-range or drift warnings occur. Unrelated guest CPU activity never exceeds 0.083%. The complete archive was independently reloaded on the VM and all 338 members verified on Windows, where all process medians and ratios were recomputed. Temporary VM settings are restored and no benchmark worker remains.

This practical measurement reports minor variation without the historical 3% A/A prerequisite. It preserves the earlier refused campaigns and makes no claim that they passed the frozen scorer.

## Evidence

[Machine-readable measurements](benchmarks-20261017.json) include all 33 raw call times for every measured worker, process medians, source/core identities, repeat spreads, process accounting, allocations, loading time and peak working set. The common runner composite SHA-256 is `67691e5c45b9ff8f5f6e2f02564ec10092bec7d0c1cb5c099ba833a6ed0eb259`; its 30-second warmup source is `b47f1f8d32c88c6f19ebd794143b13792c8b1352`.

The complete retained archive is `embeddinggemma2-practical-comparison-v59.tar.gz`, SHA-256 `547e4f06b19b5f30e81d768738988719613cc347e9b22e33262e755f7e447367`, under ignored `artifacts/embeddinggemma2/` on both machines. Its verified report SHA-256 is `2b147cc56e390df7fa803b6e4637059e6d8cb74ed54edaa180b65a31ea9d9a91`.
