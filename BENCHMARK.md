# CPU benchmarks

**Review status — 2026-09-09:** the results below are historical and must not
be used as a single-core performance baseline. The `defaults` rows compare
Lokad's one-thread default with ORT's default thread pool; their ratios use
unequal CPU resources. The `one-thread` rows disable Lokad SIMD while ORT
remains optimized. The Auto `mode-threads=1` rows match inference thread
counts, but no row enforces CPU affinity.

The `--rows canonical` selection (the default) prints only the matched
single-CPU row; `--rows all` retains the three-condition setup for diagnostics.

The replacement comparison must give **both engines the same one logical CPU**:
verified process affinity, one inference thread, Lokad Auto with available
SIMD/intrinsics, ORT CPU with intra-op 1/inter-op 1 and sequential execution,
and graph optimizations enabled. Repeat measurements on an identified, quiet
core with identical inputs, requested outputs and output-lifetime boundaries.
Scalar diagnostics and multi-core scaling belong in separate, opt-in results.
Until the harness and measurements meet that contract, retain these tables
only as dated diagnostic evidence; they cannot gate performance commits.

The model harness (`tests/Lokad.Onnx.Bench`) compares the managed engine
against native ONNX Runtime (`Microsoft.ML.OnnxRuntime` 1.23.2) sessions on
CPU only, without registering a GPU provider. The package is the CPU build;
GPU execution requires a different package/provider configuration.
[ORT C# packages](https://onnxruntime.ai/docs/get-started/with-csharp.html)

## Canonical single-CPU baseline — 2026-09-12

Measured under the contract above on LOKAD-0399 (i7-14700KF, 28 logical
CPUs): verified affinity to logical CPU 4 (efficiency-class 1,
core mask 0x30, core-group 0), one inference thread on each side, Lokad Auto with
SIMD/intrinsics, ORT CPU intra-op 1/inter-op 1 sequential with
ORT_ENABLE_ALL, High performance power scheme left unchanged. Three
independent fresh processes, 3 warmups and 33 timed iterations per engine
per case. Every case validated before and after timed reuse at the
unchanged 1e-4 gate with inputs fingerprinted intact. Machine-readable
artifacts with embedded raw samples live in
`tests/Lokad.Onnx.Bench/baseline/summary-20260912.json`, regenerated from
the per-rep logs by `python eng/parse_baseline.py <rep logs>`.
Confinement ratios were 0.97, 0.84 and 0.91 (single-threaded 2 s busy
loop; the harness fails above 1.3 and warns below 0.8): no warnings, but a
stable sub-1.00 band from box background load, so the discard rule below
still applies per session.

Measured at `8f478a3` with a clean tracked tree (local working files and
agent scratch only); no code differences. SDK
`10.0.300-preview.0.26177.108`, runtime `.NET 10.0.12`, ORT C# `1.23.2.0`,
Lokad assembly `0.2.0.0`. Asset bytes and hashes per case print in each
rep header and match `ModelManifest.json`; inputs and outputs print there
too (e5 token counts, 224x224 pixels, 4-token GPT-2 prefill).

This refresh replaces the 2026-09-11 CPU-4 table after the P65 3-row packed
GEMM campaign plus the e5 attention one-op lane additions landed; the old
table and its summary survive in git history (`summary-20260911.json` stays in
`tests/Lokad.Onnx.Bench/baseline/`).

Cells are Lokad warmed public-Execute median versus ORT warmed-Run median
per rep in milliseconds; the ratio spans the three within-rep median
ratios. Absolute medians still drift with machine settling (e5-8tok Lokad
15.1 to 23.1 ms), but both engines drift together, so cross-rep ratio
spreads hold to 0.2x on every case except bursty 8tok rows: compare revisions
within shared reps and alternate their order, never absolute medians across
days. Per-rep best, p95, max, GC, and load/prepare/first-run
figures live in the summary JSON.

| Case | rep1 L/ORT ms | rep2 L/ORT ms | rep3 L/ORT ms | Lokad / ORT |
|---|---:|---:|---:|---|
| e5-8tok | 15.1 / 6.3 | 19.6 / 7.8 | 23.1 / 7.3 | 2.4-3.2x |
| e5-30tok | 18.4 / 12.3 | 18.9 / 12.0 | 19.4 / 13.1 | 1.5-1.6x |
| dinov3-224 | 119.5 / 86.4 | 111.0 / 84.6 | 110.4 / 84.1 | 1.3-1.4x |
| resnet50-224 | 143.2 / 66.5 | 145.8 / 67.7 | 148.6 / 60.7 | 2.2-2.4x |
| gpt2-4tok | 42.8 / 29.5 | 35.0 / 25.3 | 34.6 / 24.7 | 1.4-1.5x |

Reproduce from the repo root after building Release:

```powershell
dotnet tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll e5 dinov3 resnet50 gpt2 --mode auto --threads 1 --rows canonical --cpu 4 --iters 33
```

Run it three times in fresh processes and regenerate the table with the
parser above; a rep with a confinement warning, a failed case, or a
changed asset hash is discarded and rerun, never averaged in.

## Historical methodology

Each model case reports three validated rows measured in one process:

- `defaults` — Lokad `ExecutionOptions.Default` against a default ORT
  `InferenceSession` (default thread pool, `ORT_ENABLE_ALL` graph
  optimizations).
- `one-thread` — Lokad Scalar mode (`TensorExecutionOptions.Scalar`,
  `MaxDegreeOfParallelism=1`) against ORT with `IntraOpNumThreads=1`,
  `InterOpNumThreads=1`, `ORT_SEQUENTIAL`.
- `mode-threads=N` — Lokad `--mode` with `MaxDegreeOfParallelism=N`
  against ORT with `IntraOpNumThreads=N`, `InterOpNumThreads=1`,
  `ORT_SEQUENTIAL`.

With `--threads 1`, the second and third rows share the same thread budget
but differ in kernel mode (Scalar vs the selected `--mode`, here Auto with
vectorization enabled and one thread). Keeping both documents that
distinction instead of implying a new thread condition.

Per-row isolation: each row opens its own ORT session, validates, times,
then disposes it before the next row starts, so worker threads and
optimization state from one condition do not leak into the next. The Lokad
`ComputationalGraph` is loaded once per model and `Reset()` between
executions; ORT sessions are the isolated unit. Engine order alternates per
iteration (ORT-first on even iterations, Lokad-first on odd iterations) to
avoid giving either engine a systematic warm-cache advantage.

Validation before timing: every row validates first, outside timing, with
`BenchValidate.RequireAgreement` — exact output names, float32 dtype, exact
shapes, finite values, and per-element relative tolerance 1e-4 on every
declared output against the actual session used for that row. A diverging
row throws instead of timing incorrect results.

Timing boundaries per timed iteration: the measured region is exactly one
`graph.Execute` (Lokad) or one `session.Run` (ORT) with all declared outputs
requested. Outside the region: `Reset` (reported separately as `reset`),
input conversion to `OrtValue`s (reported as `convert`), validation
executions plus comparison (reported as `validation`), and ORT output
disposal (`disposal=outside`; `using` scopes end after the stopwatch stops).
Tokenization and model parsing happen once before timing. Three warmup
executions per engine precede nine timed iterations; the tables report best
and median with p95/max in the transcript, plus the raw samples below.

No process affinity or power plan was pinned; logical processor count only,
worker spinning left at ORT defaults. Equal thread budgets do not imply
equal CPU utilization between the two engines.

## Environment — 2026-09-08 run

Runner transcript host line:

    host=LOKAD-0399 cpu=Intel64 Family 6 Model 183 Stepping 1, GenuineIntel procs=28 (logical, no affinity pinning) fma=True runtime=.NET 10.0.11 lokad=0.2.0.0 ort=1.23.2.0 ort-provider=cpu-only ort-optimizations=ORT_ENABLE_ALL mode=auto threads=1 iters=9 warmup=3

- Host: LOKAD-0399, Intel Core i7-14700KF, 20 cores / 28 logical processors.
- Build: Release, .NET SDK 10.0.300-preview.0.26177.108, .NET runtime 10.0.11.
- Source: commit `b1b447d` plus the benchmark-harness rework in this commit
  (`tests/Lokad.Onnx.Bench/Program.cs`: per-row sessions, alternated order,
  raw samples, `validation`/`convert` boundaries, output shapes, sidecars).
  Rebuilding at this commit and rerunning the command below reproduces the
  harness; numbers remain machine- and load-dependent.
- Native reference: `Microsoft.ML.OnnxRuntime` 1.23.2.0, CPU provider only,
  graph optimizations `ORT_ENABLE_ALL`, default worker spinning, no affinity.

## Assets

All models and sidecars are local; parsing and tokenization occur outside
timing. Full hashes live in
[ModelManifest.json](tests/Lokad.Onnx.Backend.Tests/ModelManifest.json).

| Asset | Model bytes | SHA-256 prefix | Sidecar | Inputs | Outputs |
|---|---:|---|---|---|---|
| multilingual-e5-small | 470,268,510 | CA456C06B3A9 | tokenizer `sentencepiece.bpe.model`, 5,069,051 bytes, CFC8146ABE2A | three int64 tensors, 1x8 or 1x30 (`input_ids`, `attention_mask`, `token_type_ids`) | `last_hidden_state` 1x8x384 / 1x30x384 |
| DINOv3 ViT-S/16 (full weights) | 137,969 | BB75E9E30FF3 | `model.onnx_data`, 86,347,776 bytes, 1EFF0BB9F4FD | float32 1x3x224x224 (`pixel_values`, filled with 0.5) | `last_hidden_state` 1x201x384, `pooler_output` 1x384 |
| ResNet50 feature export | 93,961,728 | 4F0558B775C8 | none | float32 1x3x224x224 (`input`, filled with 0.5) | `output` 1x2048 |
| GPT-2 (past-state) | 498,126,358 | 42C1E92A21C4 | none | `input_ids` 1x4, `attention_mask` 1x4, `position_ids` 1x4, 24 empty past tensors 1x12x0x64 | `logits` 1x4x50257 plus 24 present tensors 1x12x4x64 |

DINOv3 now runs against the full-weight asset (graph plus `model.onnx_data`)
and validates end to end; the earlier placeholder-asset caveat no longer
applies. DINOv2 is excluded by a tracked known-divergence condition in
`tests/Lokad.Onnx.Bench/KnownDivergences.cs`: after bit-identical
GELU fusion and tail order parity, `last_hidden_state` still diverges at reference-scaled
1.92E-004 from uniform depth-amplified fp32 summation-order drift with no localizable
kernel defect, above the unchanged 1e-4 gate. A 2026-09-12 per-layer probe (temp instrumented copy, ORT 1.29, gate metric |ref-cand|/(1+|ref|)) shows embeddings agreeing at 8e-8, the layer-0 norm output already at 1.2e-6 in ORT-vs-ORT as well, and a layer-11 attention jump to 1.48e-4 at the identical token and channel in ORT-vs-ORT and in Lokad-vs-ORT; a 1-ulp input perturbation alone moves ORT's own output by 1.0e-4 (plain) and 1.2e-4 (fused). Cross-implementation 1e-4 agreement is therefore unachievable on this operating point, while DINOv3 passes because it runs native LayerNormalization/Gelu single ops with no fusion-order differences. The case validates, reports
`case-status dinov2-224=excluded-known-divergence`, and skips every timed row, so no
DINOv2 rows are published. A breach at or above the 1e-3 tripwire, or on any
unregistered case, still fails the run as a fresh regression.

## Historical results — 2026-09-08 methodology

`Bench e5 resnet50 dinov3 gpt2 --mode auto --threads 1 --iters 9`
(three warmups, nine timed iterations per row). Median ratio is
Lokad median / ORT median. `maxdiff` is the worst validated output of that
row. Raw samples follow the tables.

| Case | Row | Lokad best | Lokad median | ORT best | ORT median | Median ratio | maxdiff |
|---|---|---:|---:|---:|---:|---:|---|
| e5, 8 tokens | defaults | 167.6 ms | 203.0 ms | 3.1 ms | 3.4 ms | 59.7x | 7.28E-007 |
| e5, 8 tokens | one-thread | 190.5 ms | 232.4 ms | 5.6 ms | 11.6 ms | 20.0x | 9.52E-007 |
| e5, 8 tokens | mode-threads=1 | 116.8 ms | 126.2 ms | 5.5 ms | 5.8 ms | 21.8x | 7.28E-007 |
| e5, 30 tokens | defaults | 201.2 ms | 260.6 ms | 4.1 ms | 5.3 ms | 49.2x | 1.15E-006 |
| e5, 30 tokens | one-thread | 405.2 ms | 430.9 ms | 12.2 ms | 12.8 ms | 33.7x | 1.24E-006 |
| e5, 30 tokens | mode-threads=1 | 144.5 ms | 154.5 ms | 11.6 ms | 13.1 ms | 11.8x | 1.15E-006 |
| DINOv3, 224x224 | defaults | 333.0 ms | 445.7 ms | 18.6 ms | 20.4 ms | 21.8x | 5.99E-006 |
| DINOv3, 224x224 | one-thread | 2193.9 ms | 2281.5 ms | 69.9 ms | 72.8 ms | 31.3x | 4.87E-006 |
| DINOv3, 224x224 | mode-threads=1 | 335.9 ms | 384.8 ms | 72.0 ms | 80.6 ms | 4.8x | 5.99E-006 |
| ResNet50, 224x224 | defaults | 245.5 ms | 281.9 ms | 7.2 ms | 8.0 ms | 35.2x | 1.98E-006 |
| ResNet50, 224x224 | one-thread | 1776.8 ms | 1788.6 ms | 50.3 ms | 51.0 ms | 35.1x | 2.53E-006 |
| ResNet50, 224x224 | mode-threads=1 | 239.0 ms | 243.7 ms | 50.5 ms | 51.0 ms | 4.8x | 1.98E-006 |
| GPT-2, 4 tokens | defaults | 745.8 ms | 990.2 ms | 8.3 ms | 9.5 ms | 104.2x | 7.35E-006 |
| GPT-2, 4 tokens | one-thread | 952.0 ms | 1010.4 ms | 23.1 ms | 27.7 ms | 36.5x | 7.95E-006 |
| GPT-2, 4 tokens | mode-threads=1 | 783.8 ms | 936.0 ms | 24.6 ms | 30.5 ms | 30.7x | 7.35E-006 |

These historical timings show gaps under each recorded condition, but the
default ratios do not measure a gap with equal CPU resources. The matched
Auto rows suggest remaining single-thread work; affinity-controlled repeat
runs are required to quantify it. The Scalar one-thread row
is much slower than the Auto one-thread row on vision models (ResNet50
1788.6 ms vs 243.7 ms median; DINOv3 2281.5 ms vs 384.8 ms), which shows the
mode distinction carries the effect, not just the thread count. Profiling
attributes ResNet50 to Conv at 90.8% (lowered through the shared GEMM
dispatcher) and e5-small to MatMul at 76.0%; generalizing the unrolled FMA
kernel to K%32 != 0 shapes moved the ResNet50 matched row from 417.6 ms to
243.7 ms with identical validation diffs.

### Raw samples (ms, n=9 per engine per row)

    e5-8tok defaults:      lok=[210.65,196.39,187.63,208.51,249.59,197.03,203.02,207.92,167.58] ort=[3.07,3.59,3.39,3.44,3.19,4.17,3.58,4.50,3.39]
    e5-8tok one-thread:    lok=[234.72,232.40,270.13,190.55,238.79,213.15,195.85,246.91,219.87] ort=[15.47,11.81,12.57,6.04,6.65,13.05,11.58,8.88,5.61]
    e5-8tok matched:       lok=[120.55,116.75,136.72,127.05,118.00,129.66,121.10,128.43,126.23] ort=[6.66,5.80,5.54,5.88,5.84,6.58,5.74,5.83,5.62]
    e5-30tok defaults:     lok=[260.62,220.43,223.90,284.85,282.43,262.30,279.52,201.18,227.03] ort=[4.75,6.19,4.69,5.71,5.29,5.74,5.00,8.55,4.10]
    e5-30tok one-thread:   lok=[570.73,553.08,475.02,421.18,419.25,407.91,454.78,430.92,405.25] ort=[27.84,16.45,13.24,12.76,12.40,12.18,13.13,12.33,12.79]
    e5-30tok matched:      lok=[145.28,148.05,160.54,146.74,144.46,155.81,154.69,163.89,154.52] ort=[13.07,35.92,12.53,12.67,11.66,13.14,11.60,14.21,14.51]
    dinov3-224 defaults:   lok=[474.40,390.33,539.31,406.05,464.43,347.45,445.66,332.98,462.18] ort=[18.98,36.59,20.41,22.00,20.04,21.29,18.84,28.46,18.63]
    dinov3-224 one-thread: lok=[2254.65,2254.56,2495.61,2430.75,2193.92,2249.55,2333.49,2281.47,2286.12] ort=[74.72,117.84,93.44,70.64,69.92,78.86,72.76,72.00,71.75]
    dinov3-224 matched:    lok=[387.11,384.81,411.62,337.38,348.35,395.20,335.89,390.08,357.03] ort=[96.41,73.04,80.70,72.02,80.56,78.83,72.95,99.71,88.74]
    resnet50 defaults:     lok=[281.87,270.54,289.81,245.50,290.84,257.56,342.69,248.34,359.53] ort=[7.22,21.58,7.43,23.46,7.50,11.27,8.01,8.11,7.69]
    resnet50 one-thread:   lok=[1793.93,1827.50,1798.57,1776.91,1814.86,1776.76,1785.98,1778.50,1788.61] ort=[51.08,50.69,50.33,51.04,51.38,50.63,51.09,51.08,50.75]
    resnet50 matched:      lok=[243.70,248.59,242.20,242.75,240.80,245.79,267.52,261.21,239.03] ort=[50.81,50.84,51.04,54.32,51.15,51.43,50.88,51.49,50.51]
    gpt2-4tok defaults:    lok=[995.01,1060.91,990.17,1363.00,1061.80,988.22,864.71,745.80,941.57] ort=[9.59,9.48,8.64,9.35,10.91,13.20,8.33,10.77,9.10]
    gpt2-4tok one-thread:  lok=[1090.96,1145.03,1066.73,984.21,1009.41,1069.98,1010.40,986.62,952.05] ort=[26.59,31.09,23.10,24.91,36.88,55.57,27.69,27.90,24.14]
    gpt2-4tok matched:     lok=[881.01,858.79,783.81,843.33,1053.93,943.16,935.97,1059.81,1376.17] ort=[48.68,37.50,27.22,27.40,24.61,30.45,26.38,38.58,37.08]

Per-row support values from the transcript: `reset` best 0.0 ms on every
row (in-place reset cost is negligible next to inference); `convert` 0.0 ms
on e5/GPT-2 rows and 0.1–0.2 ms on vision rows; `validation` 160.8–2350.8 ms
(one full Lokad execute plus one full ORT run plus comparison per output,
outside timing); `disposal=outside` throughout.

## Limits of this comparison

- One machine, one process per model set, nine samples per row. These samples
  do not establish a stable single-CPU baseline or a release performance gate.
- Running `Bench all` (or naming `dinov2`) reports the DINOv2 exclusion above
  by design and continues the remaining cases; publish tables only for validating
  models and keep the exclusion stated.
- The Lokad graph is reused across the three rows of a model with `Reset`
  between executions while ORT sessions are per-row; residual pool/cache
  effects on the Lokad side across rows are not measured separately.
- The old latency/microbenchmark tables are available in Git history; they
  predate substantial kernel and allocation changes and should not be reused
  as current measurements.

## Reproduce the historical setup

From the repository root, using the existing local assets. This command
reproduces the three-condition setup; it does not enforce the replacement
single-CPU contract described above:

```powershell
dotnet build Lokad.Onnx.slnx -c Release --tl:off --nologo -v minimal
dotnet tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll e5 resnet50 dinov3 gpt2 --mode auto --threads 1 --iters 9
```

The runner prints the host line, one `case` line per row (asset identity,
input/output shapes, warmup, iterations), one result line per row (best,
median, p95, max, reset, convert, validation, disposal, maxdiff), and one
raw-sample line per row. Keep the default, one-thread and equal-budget
results distinct; equal limits do not imply equal CPU utilization. Validate
every named output outside timing for each actual session.

For operator benchmarks, the current command is `dotnet tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll micro ops`;
`matmul2d`, `matmul` and `indexing` cover other kernel cases. The `micro oneop` lane compares five frozen one-op models
(tests/Lokad.Onnx.Bench/oneop) between a Lokad graph and a single-CPU ORT session on identical inputs, gating timing on
1e-4 agreement. Pin execution modes and record
allocations as well as latency; profiler-enabled timings are separate.

`bench.ps1` is a startup-inclusive CLI benchmark by design: it launches a
fresh CLI process for every e5 sample, so its warmup process cannot warm
those subsequent JITs/sessions. Read its `graphMs`/`wallMs` series as
per-process startup plus inference, not warmed inference throughput.
Persistent-process inference timing lives in the Bench runner above.

## Voice-model engine comparison — 2026-09-14

Measured under the single-CPU contract on ATHENA (Intel64 Family 6 Model
85, 8 logical CPUs): verified affinity to logical CPU 4 (core mask 0x30,
efficiency-class 0, SMT siblings), one inference thread on each side, Lokad
Auto with SIMD/intrinsics (Vector256, AVX-512F, FMA), ORT CPU intra-op
1/inter-op 1 sequential with ORT_ENABLE_ALL, power settings left untouched
(scheme not recorded). Three independent fresh processes, 3 warmups and 33
timed iterations per engine per case, alternating order. Every case
validated before and after timed reuse (float gate 1e-4 scaled, documented
2e-4 for the 10-second segmentation window, exact integer checks) with
fingerprinted-intact inputs. Machine-readable artifacts with embedded raw
samples live in
`tests/Lokad.Onnx.Bench/baseline/summary-voice-20260914-m7.json`,
regenerated from the per-rep logs by
`python eng/parse_baseline.py --expect parakeet-encoder,parakeet-decoder,pyannote-segmentation,pyannote-embedding <rep logs>`;
raw logs stay local under `artifacts/bench/voice-m7/`. The September 13
baseline (`summary-voice-20260914.json`, reps under
`artifacts/bench/voice-20260913/`) remains for the pre-optimization
comparison. Confinement ratios were 1.00, 0.95 and 1.00 with no warnings.

Measured at 8af223e with a clean tracked tree (local working files,
models/, and agent scratch only); no code differences across reps. SDK
10.0.102, runtime .NET 10.0.12, ORT C# 1.23.2.0, Lokad assembly 0.2.0.0.
Since the September 13 baseline the runtime gained shared-kernel LSTM
projections, Conv+Relu fusion, and bounded im2col tiling; the workload
boundaries, fixtures, and gates below are unchanged.
Graph bytes and hashes print in each rep header, including the
encoder sidecar as `sidecar=encoder-model.onnx.data` with bytes and hash:
encoder graph 41,770,866 bytes
(98A74B21B4CC) plus its
2,435,420,160-byte external weights (9A22D372C514); decoder 72,520,893
bytes (E978DDF66885); segmentation 5,916,329 bytes (AF62796ADFC4);
embedding 21,306,024 bytes (9903474D6230). Replay inputs come from the
hash-pinned `models/voice-fixtures/replay` fixtures.

| Voice case | rep1 L/ORT ms | rep2 L/ORT ms | rep3 L/ORT ms | Lokad / ORT |
|---|---:|---:|---:|---|
| parakeet-encoder | 1279.3 / 265.6 | 1260.9 / 267.1 | 1369.8 / 287.6 | 4.7-4.8x |
| parakeet-decoder | 90.9 / 8.8 | 90.3 / 8.7 | 95.9 / 9.2 | 10.3-10.5x |
| pyannote-segmentation | 217.4 / 36.6 | 216.2 / 38.3 | 233.4 / 38.3 | 5.6-6.1x |
| pyannote-embedding | 317.3 / 48.9 | 336.0 / 49.3 | 342.6 / 51.1 | 6.5-6.8x |

Cells are Lokad warmed public-Execute median versus ORT warmed-Run median
per rep in milliseconds; the ratio spans the three within-rep median
ratios. Case inputs (batch size one throughout): encoder mel [1,128,128]
to frames [1,1024,16]; decoder fixed tokens [1,5] with zero states to
scores [1,8,5,8198] plus carried states; segmentation real 10 s waveform
[1,1,160000] to scores [1,589,7]; embedding fbank [1,200,80] to frames
[1,2560,25]. Decoder and encoder integer lengths check exactly. Do not
read the fixed decoder token/frame grid or a backbone-only embedding row
as whole-model performance: decoding normally advances through individual
token/state steps, and embeddings still need masked pooling plus projection.

Startup versus warm execution (rep1): model load 2875/132/15/30 ms and
lifetime preparation 37/0.3/0.2/0.1 ms for encoder, decoder,
segmentation, and embedding against warmed medians of 1279/91/217/317
ms; first cold executions cost 2643/146/253/356 ms and are process-cold
only on the first row. Dominant remaining costs per the
post-optimization profiles are shared-matrix work in the encoder (MatMul
76%), recurrent and joint projections in the decoder (LSTM/MatMul
50/49), LSTM recurrence in segmentation (54%), and convolution in the
embedding backbone (98%). Decoder parity is the weakest at 10.3-10.5x
with no dedicated optimization this campaign; encoder parity is flat at
4.7-4.8x.

Agreement at validation (identical in all three reps): encoder
4.83e-07, decoder 6.34e-006, embedding 1.85e-006 at gate 1e-4; the
canonical real-audio segmentation row passes the standard 1e-4 gate at
1.25e-05. The synthetic 10 s fixture keeps the documented 2e-4
long-window gate in `tests/Lokad.Onnx.Bench/VoiceModelCases.cs`; the
broader long-window tolerance remains subject to numerical review. Decoder validation additionally
checks a second call with carried Lokad states supplied to both engines,
then checks reset determinism, outside timing. This does not test drift
across two independently evolving decoder trajectories.

The September 13 baseline saw segmentation Lokad medians settle
downward across reps (1444 to 1110 to 1096 ms) for an unisolated cause;
this campaign shows no such trend (217.4, 216.2, 233.4 ms) with ORT at
36.6-38.3 ms and clean confinement throughout. Rep3 ran about 5 pct slower
on both engines on the early rows (encoder ORT 287.6 ms against 265-267
ms); the within-rep ratios hold (4.76x against 4.72-4.82x), so the rep
stands. Retain the full range and compare engines within shared reps only. This host
differs from the LOKAD-0399 baseline machine, so voice ratios must not be
compared with the 2026-09-12 table. Voice keys are opt-in and excluded
from the default/`all` set (the encoder loads 2.4 GB); name them
explicitly.

Reproduce from the repo root after building Release (requires the
git-ignored voice models and replay fixtures; named cases fail loudly
when assets are absent):

```powershell
dotnet build tests/Lokad.Onnx.Bench/Lokad.Onnx.Bench.csproj -c Release --tl:off --nologo -v minimal
dotnet tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll parakeet-encoder parakeet-decoder pyannote-segmentation pyannote-embedding --mode auto --threads 1 --rows canonical --cpu 4 --iters 33
```

## Voice-model engine comparison — 2026-09-15

Measured under the single-CPU contract on ATHENA (Intel64 Family 6 Model
85, 8 logical CPUs): verified affinity to logical CPU 4 (core mask 0x30,
efficiency-class 0, SMT siblings), one inference thread on each side, Lokad
Auto with SIMD/intrinsics (Vector256, AVX-512F, FMA), ORT CPU intra-op
1/inter-op 1 sequential with ORT_ENABLE_ALL, High performance power
scheme. Three independent fresh processes, 3 warmups and 33 timed
iterations per engine per case, alternating order. Every case validated
before and after timed reuse at the standard float gate 1e-4 scaled
(segmentation passes at 1.2e-5; the 2e-4 long-window tolerance stays in
code for the historical synthetic fixture) with exact integer checks and
fingerprinted-intact inputs. Machine-readable artifacts with embedded raw
samples live in
`tests/Lokad.Onnx.Bench/baseline/summary-voice-20260915-p8.json`,
regenerated from the per-rep logs by
`python eng/parse_baseline.py --expect parakeet-encoder,parakeet-decoder,pyannote-segmentation,pyannote-embedding <rep logs>`;
raw logs stay local under `artifacts/bench/voice-p8/`. The September 14
summary (`summary-voice-20260914-m7.json`) remains for the
pre-optimization comparison. Confinement ratios were 1.00, 1.01 and 1.00
with no warnings.

Measured at de60581 with a clean tracked tree (local working files,
models/, and agent scratch only); no code differences across reps. SDK
10.0.102, runtime .NET 10.0.12, ORT C# 1.23.2.0, Lokad assembly 0.2.0.0.
Since the September 14 campaign the runtime gained decoder shared-right
batch folding; encoder persistent weight packing; LSTM prepared
transposes, vector row dots, prepared short-sequence sharing, and fused
gate dots; convolution tile alignment, rank-three and rank-four direct
depthwise paths, per-tile patch packing through the row-group kernels,
and vectorized im2col interiors; AVX512 12/8/6-row packed GEMM with
row-group dispatch including transient packs; and staged representative
voice rows. Workload boundaries, fixtures, and gates are unchanged.
Graph bytes and hashes print in each rep header and match the September
14 record: encoder graph 41,770,866 bytes (98A74B21B4CC) plus its
2,435,420,160-byte external weights (9A22D372C514); decoder 72,520,893
bytes (E978DDF66885); segmentation 5,916,329 bytes (AF62796ADFC4);
embedding 21,306,024 bytes (9903474D6230). Replay inputs come from the
hash-pinned `models/voice-fixtures/replay` fixtures.

| Voice case | rep1 L/ORT ms | rep2 L/ORT ms | rep3 L/ORT ms | Lokad / ORT |
|---|---:|---:|---:|---|
| parakeet-encoder | 734.0 / 280.6 | 682.1 / 261.9 | 705.5 / 263.7 | 2.6-2.7x |
| parakeet-decoder | 15.7 / 9.0 | 14.5 / 8.7 | 15.3 / 9.2 | 1.7-1.7x |
| pyannote-segmentation | 182.8 / 37.2 | 174.8 / 36.1 | 175.0 / 36.3 | 4.8-4.9x |
| pyannote-embedding | 176.8 / 49.5 | 185.6 / 49.1 | 176.6 / 48.5 | 3.6-3.8x |

Cells are Lokad warmed public-Execute median versus ORT warmed-Run median
per rep in milliseconds; the ratio spans the three within-rep median
ratios. Case inputs (batch size one throughout): encoder mel [1,128,128]
to frames [1,1024,16]; decoder fixed tokens [1,5] with zero states to
scores [1,8,5,8198] plus carried states; segmentation real 10 s waveform
[1,1,160000] to scores [1,589,7]; embedding fbank [1,200,80] to frames
[1,2560,25]. Decoder and encoder integer lengths check exactly. Do not
read the fixed decoder token/frame grid or a backbone-only embedding row
as whole-model performance: decoding normally advances through individual
token/state steps, and embeddings still need masked pooling plus
projection.

Startup versus warm execution (rep1): model load 3967/195/17/44 ms and
lifetime preparation 59/0.4/0.3/0.1 ms for encoder, decoder,
segmentation, and embedding against warmed medians of 734/16/183/177
ms; first cold executions cost 1556/40/287/333 ms and are process-cold
only on the first row. Dominant remaining costs per warm profiles are
encoder matrix products (289 MatMuls, about two thirds) plus
convolutions; decoder joint products and LSTM recurrence in roughly
equal halves; segmentation LSTM recurrence (about three fifths); and
embedding backbone convolution (about nineteen twentieths).

Agreement at validation (identical in all three reps): encoder
4.97e-07, decoder 4.64e-006, segmentation 1.20e-005, embedding 2.25e-006
at gate 1e-4. Decoder validation additionally checks a second call with
carried Lokad states supplied to both engines, then checks reset
determinism and independent carried trajectories, outside timing.

Repetitions interleave within about eight percent per boundary on both
engines with no settling trend; compare engines within shared reps only.
This host differs from the LOKAD-0399 baseline machine, so voice ratios
must not be compared with the 2026-09-12 table. Voice keys are opt-in
and excluded from the default/`all` set (the encoder loads 2.4 GB); name
them explicitly.

## Representative voice rows — 2026-09-15

Timed separately with the same single-CPU settings (one fresh process, 3
warmups, 33 iterations, confinement 1.00), all validating at gate 1e-4;
machine-readable summary in
`tests/Lokad.Onnx.Bench/baseline/summary-voice-rep-20260915.json`, raw
log local. These rows reuse staged replay assets only: the decoder
single steps slice the first frame and token of the step-1 fixtures
(zero states reproduce a real first decoding step; the recorded s1
states reproduce a real mid-trajectory step), encoder-64 and the
synthetic 1 s segmentation use staged fixtures, and embedding has no
second backbone fixture.

| Voice case | rep1 L/ORT ms | Lokad / ORT |
|---|---:|---|
| parakeet-encoder-64 | 458.3 / 232.9 | 2.0-2.0x |
| parakeet-decoder-1x1 | 4.4 / 3.7 | 1.2-1.2x |
| parakeet-decoder-1x1-carried | 4.4 / 3.7 | 1.2-1.2x |
| pyannote-segmentation-1s | 20.4 / 4.0 | 5.1-5.1x |

The single-step decoder runs at near parity (1.2x) with zero and carried
states alike, so the remaining bulk-grid gap sits in the joint shape,
not the step machinery. Shorter encoder audio halves the ratio (2.0x at
T=64 against 2.6-2.7x at T=128). Reproduce with
`dotnet tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll parakeet-encoder parakeet-decoder pyannote-segmentation pyannote-embedding --mode auto --threads 1 --rows representative --cpu 4 --iters 33`.

## Remaining work after 2026-09-15

Encoder parity (2.6-2.7x) is limited by large-matrix kernel efficiency
against native 12-row kernels; reduction blocking, software prefetch,
and pointer tuning were measured and rejected on full-workload
evidence. The decoder bulk grid (1.7x) contrasts with single-step parity
(1.2x): the joint shape, not per-step work, dominates. Segmentation
(4.8-4.9x) is limited by small-matrix recurrent efficiency. Embedding
(3.6-3.8x) still needs cross-layer blocked-channel layout with residual
fusion; per-tile patch packing, vectorized im2col, and single-layer
blocked prototypes were measured, with only the first two promoting.
Longer-encoder replay is unstaged, matched TorchSharp/ORT single-step
calibration and encoder activation fusion are outstanding, and the
managed-dependency smoke (no TorchSharp, LibTorch, or ORT in the
published closure, 9/9 replay checks) must be re-run after further
runtime changes.
