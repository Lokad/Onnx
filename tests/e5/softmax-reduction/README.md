# Exact masked-softmax maximum reduction prototype

This standalone prototype changes the final maximum/NaN reductions in the
existing paired-row masked softmax. It has no product dispatch and no measured
speed claim. See [the retained local proof](results-20260919.md).

`generate.py` extracts the original helper and full paired-row kernel from
`TensorOps.Elementwise.cs`, using the existing balanced-brace extractor. It
removes the three unselected wide-exponential branches and retains the original
nonpositive exponential. The candidate preserves vector accumulation, scalar
tails and comparison order. For eight-lane vectors it replaces the variable
index maximum loop with seven literal lane comparisons. It replaces the
validity-mask loop with `Vector.EqualsAll`; validity lanes are all-zero or
all-one. The original maximum loop remains for other vector widths.

`Program.cs` binds typed delegates to the actual frozen product methods and
compares both the copied and candidate implementations with them. It exercises
random float bits, exceptional values and NaN payloads, signed zeros, masked
rows, offsets, empty/odd/even shapes, scalar/SIMD paths, guarded destinations,
unchanged inputs, in-place output and short-mask refusal. Finite complete
outputs are also compared with independent double normalization. This does not
replace full-model numerical or ownership qualification.

The exact core is
`187de61ad8f034b9b7ad2fb3490358443fa84334204720e81bc3546a31f3c8d4`.
From the repository root, choose fresh destinations and run:

```powershell
python -B tests/e5/softmax-reduction/generate.py --output artifacts/reduction-new/generated
dotnet build tests/e5/softmax-reduction/Probe.csproj -c Release -o artifacts/reduction-new/bin -p:FrozenCorePath=C:/Users/JoannesVermorel/code/Onnx/artifacts/softmax-zero-product-20260919/frozen/Lokad.Onnx.dll -p:KernelSourceDirectory=C:/Users/JoannesVermorel/code/Onnx/artifacts/reduction-new/generated --tl:off --nologo -v minimal
artifacts/asr-labeled-20260919/venv/Scripts/python.exe -B tests/e5/softmax-reduction/run_check.py --binary artifacts/reduction-new/bin/Probe.dll --output artifacts/reduction-new/check --cpu 4
```

The supervisor needs `psutil` and refuses existing evidence directories. It
inherits CPU affinity before CLR startup, uses normal runtime settings plus
code capture, and bounds the child to 1 GiB and 60 seconds. Conditioning yields
periodically to other runtime threads; neither conditioning duration nor job
duration is an inference benchmark. Use a valid logical CPU on the target host.

The local result establishes exactness for the exercised eight-lane/scalar
paths and removal of repeated stores in captured optimized loop code. It does
not establish other vector widths, final Tier1 code, AMD speed, or a complete
model benefit. An AMD experiment requires a separately frozen protocol after
the active zero-block product comparison closes. The earlier E82 pointer
experiment remains closed; this prototype changes different instructions.
