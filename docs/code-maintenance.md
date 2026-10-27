# Development guide

## Project boundaries

The solution targets .NET 10. The core `src/Lokad.Onnx` project is the only
NuGet package and has no runtime dependencies. It contains tensors, CPU kernels,
the neutral `OnnxModel` description, graph preparation and execution.

The nonpackable [Protobuf adapter](../tools/Lokad.Onnx.Protobuf/README.md)
implements `IOnnxModelReader` and owns generated messages and parser dependencies.
[Data](../src/Lokad.Onnx.Data/README.md) provides text/image/audio workflows;
[CLI](../src/Lokad.Onnx.CLI/README.md) exposes them as `lonnx` commands. Tests,
benchmarks and model/reference tools are also nonpackable. Keep parser types and
companion dependencies out of the core's public boundary.

## Build, test and pack

Use the SDK selected by global.json and PowerShell 7. From the repository root:

```powershell
pwsh -NoProfile -File eng/restore-tokenizers.ps1
pwsh -NoProfile -File eng/test.ps1 -Configuration Release -BuildTools
pwsh -NoProfile -File eng/smoke-pack.ps1 -Configuration Release
```

The restore script builds the pinned Lokad.Tokenizers source package and verifies
the consumed dependency. A cache mismatch fails explicitly; use a fresh
`NUGET_PACKAGES` directory rather than modifying a shared cache.

`eng/test.ps1` restores, builds and runs `eng/Tests.slnx`, including CLI integration
tests. `-BuildTools` also builds the full root solution; `-Coverage` opts into
local coverage. To reuse a Release build:

```powershell
dotnet test eng/Tests.slnx -c Release --no-build --tl:off --nologo -v minimal
```

Set `DOTNET_EnableHWIntrinsic=0` for the scalar/fallback gate. Always disable the
.NET terminal logger with `--tl:off`. Ordinary tests use bundled fixtures;
large-model tests require explicitly staged local assets and report missing
assets as skips. Use the existing local e5 model at
models/multilingual-e5-small/model.onnx. See the [testing guide](../tests/README.md)
for strict local-model selection and Python/reference setup.

`pack.cmd` creates the Release core package. `eng/smoke-pack.ps1` also restores
and runs a consumer using only that package from an isolated local feed, checking
contents and the absence of runtime dependencies.

## Correctness and optimization

`CPUExecutionProvider.SupportedOps` and `OperatorSchemas.All` define supported
operators and schema rules. Preserve exact operator names, opsets, shape/type
validation, input immutability and [output lifetime](runtime-options.md).
Views require root-storage alias checks; unknown aliases must remain conservative.
Public APIs can have external users even when repository searches find few calls.

Keep independent expectations in tests/Shared/ReferenceKernels; a reference must
not call the optimized implementation it checks. `eng/test-opfuzz.ps1` runs the
single-op native differential lane. `eng/test-e5.ps1`,
`eng/test-embeddinggemma2.ps1`, `eng/test-whisper.ps1` and
`eng/test-parakeet.ps1` cover selected models with staged assets. Model-specific
frontend and public-result checks are linked from the [support guide](model-support.md).
Numerical tolerance, decision agreement, ownership and accuracy are separate checks.

At fixed execution mode and thread count, compare deterministic repeats; do not
demand bitwise identity across modes. The e5 lane disables JIT on-stack
replacement for deterministic children. Exceptional-value contracts distinguish
NaN classification from unspecified NaN payload/sign selection in independently
compiled arithmetic. Keep original input bits intact. Measure allocation on a
warmed dedicated thread and use the nonparallel allocation collection when
checking exact zero-allocation contracts.

Use [eng/bench.py](../eng/bench.md) for routine CPU comparisons and
[BENCHMARK.md](../BENCHMARK.md) for published measurements and protocols. Run one
timing lane at a time, clear verified owned leftover processes, keep all samples
and compare identical inputs. A microbenchmark gain alone does not qualify a
model-wide speedup. Similar arithmetic can need separate compiled entrypoints
for different JIT attributes or vector widths; validate affected models before
consolidating kernels. New runtime branches need a concrete workload and an
acceptance criterion.

## CI

Every push/PR classifies changed paths. Markdown-only changes skip the numerical
matrix; packaged README/CHANGELOG changes still run package smoke. Code, package
metadata, workflows, fixtures and unknown paths conservatively run the full
Windows/Linux ordinary/scalar, differential and package jobs. Deletions and
renames count. The Linux ordinary job also builds all solution tools.

After a push, wait at least **180 seconds** before checking CI and between status
observations; increase the interval while queued. Investigate failures rather
than probing repeatedly or adding automatic retries.
