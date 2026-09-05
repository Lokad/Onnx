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

- `src\Lokad.Onnx.Data`: text + image IO helpers (tokenizers, ImageSharp).
- `src\Lokad.Onnx.CLI`: `lonnx` CLI (verbs: `info`, `run`, `benchmark`); output in `src\Lokad.Onnx.CLI\bin\Release\net10.0\`.

- `tests\Lokad.Onnx.Backend.Tests`: xUnit backend tests + MNIST assets.
- `tests\Lokad.Onnx.Tensors.Tests`: xUnit tensor tests.
- `tests\e5`: native ONNX Runtime e5 conformance (`eng/test-e5.ps1`); `tests\opfuzz`: single-op differential lane (`eng/test-opfuzz.ps1`).
- `ext\NLog`: vendored logging source, not referenced by the solution.
- `ext\satsumagraph`: removed; no submodule or graph-algorithm dependency remains.

## Build / run / test (repo conventions)
- Build all: `build.cmd` (restore + CLI build). Uses `--tl:off`.
- Pack NuGet: `pack.cmd` (builds `src\Lokad.Onnx`, the packable core).
- CLI wrapper: `lonnx.cmd` calls Release `Lokad.Onnx.CLI.exe`.
- Run tests: `dotnet test --tl:off --nologo -v minimal --no-build` from repo root.
- Python lanes: `pip install -r tests/e5/python/requirements.txt`, then `eng/test-e5.ps1` or `eng/test-opfuzz.ps1` (run under pwsh 7).

## CLI quick notes
- `lonnx info <model.onnx>`: model metadata; supports `--ops`, `--init`, `--op-filter`.
- `lonnx run <model.onnx> <inputs...>`: supports image/text inputs, `--softmax`, `--print-input`, profiling, SIMD toggles.
- `lonnx benchmark <id>`: `matmul2d`, `matmul`, `indexing`, `me5s-load`, `me5s-run` with BenchmarkDotNet flags.
