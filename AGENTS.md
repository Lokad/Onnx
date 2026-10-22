## Turn off the .NET “terminal logger”

Use `--tl:off` avoids dynamic output and progress rendering.

```powershell
dotnet restore --tl:off -v minimal
dotnet build   --tl:off --nologo -v minimal
dotnet test    --tl:off --nologo -v minimal --no-build
```

# ExecPlans
 
When writing complex features or significant refactors, use an ExecPlan (as described in .agent/PLANS.md) from design to implementation.

## Local models

The ONNX model at `models\multilingual-e5-small\model.onnx` is already downloaded from Hugging Face and is git-ignored. Use it for the `lonnx run` e5 example instead of fetching the URL.

## Repo map (high-signal)
- `Lokad.Onnx.slnx`: root solution (XML format); targets net10.0 across projects.
- `src\Lokad.Onnx`: core library and NuGet package (tensors, graph execution, shared infra); unsafe + SIMD options.

- `src\Lokad.Onnx.Data`: nonpackable text/image IO, embedding and audio APIs (tokenizers, ImageSharp).
- `src\Lokad.Onnx.CLI`: `lonnx` CLI (verbs: `info`, `run`, `embed`, `diarize`, `transcribe`); output in `src\Lokad.Onnx.CLI\bin\Release\net10.0\`.

- `tests\Lokad.Onnx.Backend.Tests`: xUnit backend tests + MNIST assets.
- `tests\Lokad.Onnx.Tensors.Tests`: xUnit tensor tests.
- `tests\e5`: native ONNX Runtime e5 conformance (`eng/test-e5.ps1`); `tests\opfuzz`: single-op differential lane (`eng/test-opfuzz.ps1`).
- `ext\satsumagraph`: removed; no submodule or graph-algorithm dependency remains.

## Build / run / test (repo conventions)
- Build all: `build.cmd` (restore + CLI build). Uses `--tl:off`.
- Pack NuGet: `pack.cmd` (builds `src\Lokad.Onnx`, the packable core).
- CLI wrapper: `lonnx.cmd` calls Release `Lokad.Onnx.CLI.exe`.
- Run ordinary tests: `eng/test.ps1 -Configuration Release` restores, builds and tests `eng/Tests.slnx`. Add `-BuildTools` to also compile all solution tools.
- Reuse a Release build: `dotnet test eng/Tests.slnx -c Release --tl:off --nologo -v minimal --no-build`. Set `DOTNET_EnableHWIntrinsic=0` for the hardware-disabled gate.
- Python lanes: `pip install -r tests/e5/python/requirements.txt`, then `eng/test-e5.ps1` or `eng/test-opfuzz.ps1` (run under pwsh 7).

## CLI quick notes
- `lonnx info <model.onnx>`: model metadata; supports `--ops`, `--init`, `--op-filter`.
- `lonnx run <model.onnx> <inputs...>`: supports image/text inputs, `--softmax`, `--print-input`, profiling, SIMD toggles.
- `lonnx embed`, `diarize` and `transcribe` use prepared local assets; see the [CLI guide](src/Lokad.Onnx.CLI/README.md) and [Data APIs](src/Lokad.Onnx.Data/README.md).
- Model comparisons use `eng/bench.py` and committed prepared cores; follow [eng/bench.md](eng/bench.md), with [Gemma details](tests/embeddinggemma2/README.md#cpu-benchmarks). The older `bench.ps1` includes process startup in its measurements.
- Microbenchmarks live in `tests/Lokad.Onnx.Bench`: `dotnet Lokad.Onnx.Bench.dll micro <matmul2d|matmul|indexing|ops>` with BenchmarkDotNet flags.

## CI

Anonymous GitHub API calls are capped at 60/hour: wait at least 180 seconds after a push and between CI status observations, increasing intervals while queued. Never poll in a tight loop.
