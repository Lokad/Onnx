# 0.4.0 CPU benchmark refresh — 2026-10-27

Measured source: `83f4e1b43656faad42c32e2efcae4e5b2cc3c0a0`, built with SDK
10.0.204, executed on .NET 10.0.8. Host: Azure Standard_D4as_v7, AMD EPYC 9V45,
4 vCPU / 16 GiB; CPU 2 compute, CPU 0 supervision. Native inference is ORT 1.29.0,
single-thread sequential CPUExecutionProvider with graph optimizations enabled.

All seven workloads pass their native/managed correctness checks and all 28
timing workers complete successfully. The [machine-readable evidence](release-0.4.0-20261027.json)
retains every warmup/measured clock, process interval, peak sampled RSS, input,
model and binary identity, numerical results and the complete aggregate calculation.
The source additionally passes 5,742 ordinary Release tests / 37 existing skips,
100 focused Community-1/symbol checks, 71 geometry/Slice checks and package smoke.

## Repeated process results

| Model | Lokad process results, s | ORT process results, s |
|---|---:|---:|
| Parakeet TDT 0.6B V3 | 33.620406 / 33.097689 | 17.945357 / 17.837115 |
| Pyannote Community-1 | 6.953607 / 7.053558 | 3.951952 / 3.983840 |
| multilingual-e5-small | 0.010847 / 0.010861 | 0.009396 / 0.009585 |
| EmbeddingGemma 2 (FP32) | 0.177488 / 0.176337 | 0.179377 / 0.178289 |
| DINOv3 ViT-S/16 | 0.060107 / 0.059985 | 0.050508 / 0.050507 |
| ResNet50 | 0.066352 / 0.067883 | 0.032340 / 0.032366 |
| GPT-2 | 0.026225 / 0.026776 | 0.018061 / 0.018248 |

The root [benchmark table](../../BENCHMARK.md) reports the median of each pair.
Each graph process result is the mean of its 180 measured calls; e5 has 1,200
warmups and the others 600. Each audio result is the mean of three measured calls
per fixture after one warmup; Parakeet sums all twenty clip means, Pyannote selects
dialogue-30s from its four checked fixtures. Each Gemma result is the median of
33 calls after the existing >=32-call / 10-second steady warmup. Process order is
managed/native/native/managed for every workload. No clocks are discarded.
Ranges describe observed variation, not statistical confidence intervals or
acceptance by the former strict A/A scorer. These timings replace older-host,
older-source figures; their differences do not isolate a single optimization.

## Correctness and release fixes

Graph workers compare complete outputs against frozen references and an independent
aggregation compares newly produced managed/native arrays directly, requiring
exact shapes, finite values and maximum scaled error <=1e-4. Input and held-output
checks remain active. Audio workers check every normalized public result, exact
token/frame/duration decisions or speaker timelines, centroids at the 1e-4 bound,
input immutability and all held results. This does not resolve the broader
[audio tensor/accuracy limitations](../../docs/model-support.md#numerical-limits).
Gemma retains component allowance 1e-6+1e-5*abs(reference), relative L2 <=1e-4,
cosine >=0.99999 and unit-norm error <=1e-5, before and after timing.

Initial checks found two release compatibility issues, fixed before final timing.
The reverse-Slice marker -Int64.MaxValue used by both audio exports is now
accepted alongside the other explicit markers, with ordinary geometry still
bounded at 2^48. The pinned segmentation export reused frames for a pre-downsampled
branch and final output of different lengths. [ONNX dimension symbols are global](https://onnx.ai/onnx/repo-docs/IR.html#static-tensor-shapes),
including subgraphs; its Data loader corrects only that exact file's branch
annotation. Core validation, model operators and weight bytes stay intact.
Initial failed/partial workers are retained separately and contribute no final
timing samples. A taskset startup-observation race was corrected by setting CPU
affinity before exec; all final worker-thread affinity observations are CPU 2.

## Reproduction and identities

Graph and audio inputs are the retained September fixtures, with exact hashes
in the JSON; Gemma uses benchmark-fixtures-v1 and the pinned prepared FP32 export.
Graph input tensors are frozen, including 0.5-valued 224x224 images and four-token
GPT-2 input with empty past state. The original graph/audio consumers are restored
from archive commit f11777fb91eeff930db8d5a545ffe3481e9f517f:
tests/benchmarks/release-amd-v2/Program.cs and tests/audio/comparison/Program.cs.
Rebuild them against the measured core, Data and separate Protobuf adapter.
The graph consumer's prospective timing configuration is 1,200 e5 / 600 other
warmups plus 180 measured calls; its validation and Reset/Execute/owned-array
boundary are unchanged. Native graph and audio workers use the corresponding
retained native.py/audio_adapter.py, pinned ORT/frontend dependencies and the
same fixtures. Gemma uses current tests/Lokad.Onnx.EmbeddingGemma2Bench with a
576 MiB packed-weight budget and 768 dimensions. No model download or production
runtime switch is part of the measurement.

Measured core SHA-256: `07f6b3987aa674434172f040c83fd515ad5cc00b3dec4226d2d6b5ac6c547c3e`.
Data SHA-256: `32e8af29f959595194f0a0d502b77673b6333586aa5be6957867a586d8a0f8b2`.
The JSON also records adapter, worker and native-library identities. Core bytes
match the ordinary-test binaries; source code is committed before measurement.

DINOv2 was rechecked untimed with the same current core and ORT. Its maximum
scaled difference is 0.000148850995, beyond 1e-4; no timing is
published. See the JSON exclusion record. Whisper is outside this seven-figure
refresh and remains without a matched timing.
