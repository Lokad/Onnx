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
