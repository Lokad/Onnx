# Exact zero-block softmax prototype

This standalone experiment skips the exponential polynomial only when every
computed vector lane is below the existing `-88.722839f` cutoff. It preserves
the existing zero result, reduction order, division, scalar tails and NaN path.
The adaptive wrapper scans the shared mask once and selects the original loop
if no mask element is below the cutoff. The mask scan never proves the skip:
each skipped vector must pass the check on its actual exponential arguments.

The [September 19 result](results-20260919.md) passes correctness but fails the
prospective timing screen. No product route or default was added.

`generate.py` extracts the current nonpositive-on, wide-off implementation,
specializing only those qualified defaults. It asserts transformation counts
and records source hashes. The probe checks a copied control and both candidate
routes against the actual frozen core method bound through a delegate.

From the repository root in PowerShell, use a fresh output directory:

```powershell
python -X utf8 tests/e5/softmax-zero-blocks/generate.py --output artifacts/softmax-zero-next/kernels
$zeroCore = (Resolve-Path artifacts/production-defaults-v2-20260919/frozen/Lokad.Onnx.dll).Path
$zeroKernels = (Resolve-Path artifacts/softmax-zero-next/kernels).Path
dotnet build tests/e5/softmax-zero-blocks/Probe.csproj -c Release --tl:off --nologo -v minimal -p:FrozenCorePath=$zeroCore -p:KernelSourceDirectory=$zeroKernels -o artifacts/softmax-zero-next/bin
dotnet artifacts/softmax-zero-next/bin/Probe.dll artifacts/softmax-zero-next/local-check.json check
python -X utf8 tests/e5/softmax-zero-blocks/test_audit.py
```

The core SHA must be
`7653c1686419d612e2624740908a44ffdee91b4239628a979bac44f5b6b863e9`.
The recorded build used SDK 10.0.204. The local check verifies 2,503,712
exponential lanes, 918 complete tensors and four short-mask refusals on an
eight-lane host. It writes a new JSON result and refuses an existing file.

`prepare.py` freezes the small payload after the local check. Its inputs include
the focused plan at `.agent/m2-softmax-zero-blocks-20260919.md`; the historical
campaign's frozen copy is retained inside its payload. Do not use that closed
plan as authorization to repeat the same campaign. A distinct follow-up needs
its own prospective protocol and fresh artifact directory.

`run.py` runs on Linux: eight fresh CPU-2 workers under ordinary tiering/GC,
four mode rotations and their reversals, plus one separate disassembly worker.
The supervisor runs on CPU 0, enforces 180 seconds and 1 GiB per child, verifies
process birth before stopping an owned process, and records process/CPU/GC
observations. It requires Python's standard library and the frozen
`eng/campaign_processes.py` helper. Each worker conditions every shape/mode
before collecting any measurements. No JSON, hash or console work occurs
between measured batches. All sample storage is preallocated.

After terminal verification and digest-checked collection, independent audit is:

```powershell
python -X utf8 tests/e5/softmax-zero-blocks/audit.py --artifact artifacts/softmax-zero-next --output artifacts/softmax-zero-next/audit.json
python -X utf8 tests/e5/softmax-zero-blocks/export.py --artifact artifacts/softmax-zero-next --output artifacts/softmax-zero-next/observations.json
```

The artifact layout must contain `payload`, `bundle.tar.gz`, `results.tar.gz`,
and the extracted remote tree under `collected`, including `collection.json`.
The latter binds every collected file and the independently verified terminal
process identities. A timing-screen failure is written explicitly in the audit;
the audit's zero exit code means the evidence was valid, not that performance
passed. Existing successful artifacts and evidence writers must not be rerun.
