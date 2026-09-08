# CPU benchmarks

The model harness uses `Microsoft.ML.OnnxRuntime` 1.23.2 and constructs CPU
sessions without registering a GPU provider. The package is the CPU build;
GPU execution requires a different package/provider configuration.
[ORT C# packages](https://onnxruntime.ai/docs/get-started/with-csharp.html)

The previous model table compared single-threaded Lokad with ORT's default
CPU thread pool, using an earlier implementation and a first-output harness.
It was not an equal-thread comparison. ORT's default intra-op pool uses physical
CPU cores and enables graph optimizations and worker spinning. This does not
establish that every operator uses every core, or that threading and matrix
packing explain most of the observed gap.
[ORT thread settings](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)

## Diagnostic run — 2026-09-08

Source: `a5c7aba7be9a46392625ee69a49db37eeaa07ac1`, Release build.
Host: LOKAD-0399, Intel Core i7-14700KF, 20 cores / 28 logical processors,
x86 FMA available; .NET runtime 10.0.11; ORT assembly 1.23.2.0.
Build SDK: 10.0.300-preview.0.26177.108.

Three warmups and nine timed iterations per row, interleaved Lokad then ORT.
Inputs and models are local; parsing and tokenization occur outside timing.
The runner measures Execute/Run with all declared outputs requested. Explicit
Reset, input conversion and ORT output disposal are outside the timed region.
No process affinity or power configuration was pinned for this run.

Default settings: Lokad Auto, one thread; ORT default session options.

| Case | Lokad best | Lokad median | ORT best | ORT median | Median ratio |
|---|---:|---:|---:|---:|---:|
| e5, 8 tokens | 155.8 ms | 201.1 ms | 2.9 ms | 3.2 ms | 62.8× |
| e5, 30 tokens | 207.2 ms | 231.4 ms | 4.1 ms | 4.7 ms | 49.2× |
| ResNet50, 224×224 | 344.5 ms | 409.0 ms | 8.0 ms | 15.2 ms | 26.9× |

Explicit one-thread settings: Lokad Auto with MaxDegreeOfParallelism=1;
ORT IntraOpNumThreads=1, InterOpNumThreads=1, ORT_SEQUENTIAL.

| Case | Lokad best | Lokad median | ORT best | ORT median | Median ratio |
|---|---:|---:|---:|---:|---:|
| e5, 8 tokens | 120.9 ms | 124.2 ms | 5.9 ms | 6.5 ms | 19.1× |
| e5, 30 tokens | 150.0 ms | 154.7 ms | 11.9 ms | 13.2 ms | 11.7× |
| ResNet50, 224×224 | 276.1 ms | 286.3 ms | 50.6 ms | 51.6 ms | 5.5× |

The one-thread rows still show a substantial CPU performance gap. The old
~199× ResNet figure is not a current equal-thread baseline. These samples also
show Lokad latency changing between the two lanes despite equivalent Lokad
settings: session interference, run order, JIT/GC and host activity need to be
controlled before attributing changes to kernels or thread counts. In
particular, the slower e5 figures do not establish a regression magnitude
against the historical table without a controlled before/after run.

Assets are identified in
[ModelManifest.json](tests/Lokad.Onnx.Backend.Tests/ModelManifest.json):

| Asset | Model bytes | SHA-256 prefix | Inputs | Output |
|---|---:|---|---|---|
| multilingual-e5-small | 470,268,510 | CA456C06B3A9 | three int64 tensors, 1×8 or 1×30 | last_hidden_state |
| ResNet50 feature export | 93,961,728 | 4F0558B775C8 | float32, 1×3×224×224, filled with 0.5 | output |

The manifest records full hashes, including the e5 tokenizer. This run did
not refresh DINOv2, DINOv3 or GPT-2 timing. DINOv3 currently fails its
LayerNormalization gate and its recorded asset contains placeholder-sized
initializers; it needs a full-weight asset and validated outputs before a
model-speed claim.

## Limits of the current harness

The source review found issues that must be fixed before treating this as a
validated performance baseline:

- Validation compares flattened lengths, omits shape equality, and can accept
  NaN differences. The reported maximum errors (7.28e-7, 1.15e-6 and 1.98e-6)
  come from that existing check against the default ORT session; the exact
  matched-thread ORT session is not validated.
- Graph Conv currently drops explicit execution options. Its Auto/one-thread
  fallback agrees with the setting requested above, but other mode/thread
  comparisons cannot rely on that route until corrected.
- Metadata probe sessions and the default ORT session remain alive during
  portions of the comparison. Worker spinning can affect nearby Lokad samples.
  The engine order is always Lokad first.
- The printed `copy` value includes two full validation executions and comparison
  work. It is not isolated copy time. The runner reports summary statistics,
  but does not retain all raw samples or explicit optimization/spinning settings.

Resolve these issues before publishing a release performance baseline. The old
latency/microbenchmark tables are available in Git history; they predate
substantial kernel and allocation changes and should not be reused as current
measurements.

## Reproduce and extend

From the repository root, using the existing local assets:

```powershell
dotnet build Lokad.Onnx.slnx -c Release --tl:off --nologo -v minimal
dotnet tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll e5 resnet50 --mode auto --threads 1 --iters 9
```

The runner prints both default and explicit-thread lanes. Once option routing
and validation are repaired, repeat with a chosen common thread budget
(`--threads N`) and sufficient samples; equal limits do not imply equal CPU
utilization. Keep the default, one-thread and equal-budget results distinct.
Validate every named output's dtype, shape, finite values and numerical
tolerance outside timing for each actual session.

For operator benchmarks, the current command is `dotnet tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll micro ops`;
`matmul2d`, `matmul` and `indexing` cover other kernel cases. Pin execution modes and record
allocations as well as latency; profiler-enabled timings are separate.

`bench.ps1` launches a fresh CLI process for every e5 sample. Its separate
warmup process cannot warm those subsequent JITs/sessions; interpret it as
startup-inclusive CLI measurement, not warmed inference throughput.
