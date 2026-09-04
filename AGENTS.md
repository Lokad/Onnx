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
- `Lokad.Onnx.slnx`: root solution (XML format); targets net10.0 across projects; includes vendored `ext\NLog`.
- `src\Lokad.Onnx.Base`: shared infra (logging, profiling, hardware config, utilities).
- `src\Lokad.Onnx.Tensors`: core tensor types + ops (managed port of ONNX Runtime tensors); unsafe + SIMD options.
- `src\Lokad.Onnx.Backend`: ONNX graph execution (Model/Node/ComputationalGraph/CPUExecutionProvider).
- `src\Lokad.Onnx.Data`: text + image IO helpers (tokenizers, ImageSharp).
- `src\Lokad.Onnx.CLI`: `lonnx` CLI (verbs: `info`, `run`, `benchmark`); output in `src\Lokad.Onnx.CLI\bin\Release\net10.0\`.
- `src\Lokad.Onnx.Interop`: .NET assembly used by Python interop (`src\interop`).
- `src\interop`: Python bridge + helpers; used by `tests\python` (requires pythonnet + transformers + onnx).
- `src\Lokad.Onnx.Package`: NuGet pack that compiles Base/Tensors/Backend into single `Lokad.Onnx` package.
- `tests\Lokad.Onnx.Backend.Tests`: xUnit backend tests + MNIST assets.
- `tests\Lokad.Onnx.Tensors.Tests`: xUnit tensor tests.
- `tests\python`: pytest suite comparing .NET vs NumPy/transformers.
- `ext\NLog`: vendored logging source; solution includes project reference.
- `ext\satsumagraph`: removed; no submodule or graph-algorithm dependency remains.

## Build / run / test (repo conventions)
- Build all: `build.cmd` (restore + CLI build + interop publish). Uses `--tl:off`.
- Pack NuGet: `pack.cmd` (builds `src\Lokad.Onnx.Package`).
- CLI wrapper: `lonnx.cmd` calls Release `Lokad.Onnx.CLI.exe`.
- Run tests: `dotnet test --tl:off --nologo -v minimal --no-build` from repo root.
- Python tests: build interop first (`build.cmd`), then `pip install -r src\interop\requirements.txt`, run `pytest` from repo root.

## CLI quick notes
- `lonnx info <model.onnx>`: model metadata; supports `--ops`, `--init`, `--op-filter`.
- `lonnx run <model.onnx> <inputs...>`: supports image/text inputs, `--softmax`, `--print-input`, profiling, SIMD toggles.
- `lonnx benchmark <id>`: `matmul2d`, `matmul`, `indexing`, `me5s-load`, `me5s-run` with BenchmarkDotNet flags.
