# Wider LayerNorm final-transform prototype

This standalone prototype keeps both centered, double-precision statistics
passes unchanged. It widens only `((x - mean) * inv) * scale + bias`, then narrows
to float. The original remaining-vector and scalar tails are retained. No
production dispatch, dependency or default changes.

`generate.py` extracts the entire current `LayerNormFloatInto` body. `Copy` is
the original; `WideOutput` inserts a sixteen-float loop using two512-bit double
vectors. The proof also calls the actual private kernel from the qualified
archived product through a typed delegate. That actual implementation is the
comparison reference, not just another generated copy.

The [local proof](results-20260920.md) includes all125 LayerNorm instances from
five full e5 cases, complete native model output checks, and synthetic shape,
tail, optional-bias, epsilon, exceptional-float and random-bit cases. Every
tested output bit, input/parameter guard and in-place result must match.
Independent Python checks all saved real arrays against scalar centered-double
normalization and reconstructs every synthetic input. Local512-bit operations
use software fallback; actual AMD code and speed remain unqualified.

From the repository root, choose a fresh artifact directory and run:

```powershell
C:/Python313/python.exe -X utf8 -B tests/e5/layernorm-output/generate.py --output artifacts/e5-layernorm-output-next/generated
dotnet build tests/e5/layernorm-output/Probe.csproj -c Release --tl:off --nologo -v minimal -p:FrozenProductDirectory=C:/Users/JoannesVermorel/code/Onnx/artifacts/e5-fingerprint-product-v2-20260920/payload/tests/e5/fingerprint-product/bin/Release/net10.0 -p:KernelSourceDirectory=C:/Users/JoannesVermorel/code/Onnx/artifacts/e5-layernorm-output-next/generated -o artifacts/e5-layernorm-output-next/bin-final
C:/Python313/python.exe -X utf8 -B tests/e5/layernorm-output/run.py --artifact artifacts/e5-layernorm-output-next --mode capture
C:/Python313/python.exe -X utf8 -B tests/e5/layernorm-output/run.py --artifact artifacts/e5-layernorm-output-next --mode proof
C:/Python313/python.exe -X utf8 -B tests/e5/layernorm-output/audit.py --artifact artifacts/e5-layernorm-output-next --output artifacts/e5-layernorm-output-next/audit.json
```

The evidence-specific refusal tests and closer bind the retained20260920
artifact and its diagnostic records. Do not invoke them against an unrelated
future run without declaring that run's scope and identities. Successful
writers refuse existing outputs. Workers inherit CPU2; guards are180seconds,
6GiB group RSS and2GiB available memory. Child-only environment cleanup preserves
global settings and unrelated work. No model downloads or native ORT inference.

A future AMD experiment must first establish actual AVX512 output equality and
generated instructions, then time complete normalization banks including
statistics and tails. A faster final loop alone cannot establish a model gain.
