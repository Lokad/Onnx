# AMD masked-softmax reduction comparison

This additive experiment measures the complete kernel from the
[closed local exactness proof](../softmax-reduction/results-20260919.md).
It reuses the corrected long-batch measurement routine and supervision from
`softmax-batch-control`, with a new protocol and candidate. No product method,
default, old experiment or active campaign changes.

The frozen core SHA is
`187de61ad8f034b9b7ad2fb3490358443fa84334204720e81bc3546a31f3c8d4`.
Both original and candidate are extracted by `softmax-reduction/generate.py`.
The candidate preserves vector accumulation and ordered maximum comparisons,
replacing final variable lane reads by literal indices and the validity loop
by one vector test. It does not include zero-block skipping.

Every worker first checks 109,488 maxima, 1,728 complete tensors, 3,456 in-place
outputs and three refusals against the actual frozen product. Finite outputs
also satisfy independent double normalization at `1e-6` absolute error.

The four timing slots are actual, copied, duplicate and probe. Copied and
duplicate use the same original delegate. Control-phase probe is a third
reference to it; comparison-phase probe uses the reduced candidate. Eight
fresh sequential AMD CPU-2 workers run four rotations and their reversals per
phase. Supervisor affinity is CPU 0; runtime is normal .NET 10.0.8.

Shapes 8/30/padded128/128/512/padded512 have twelve times as many rows as
columns. Batches use 32768/4096/256/256/32/32 calls respectively. Every
shape/slot is conditioned for at least one second through the exact measurement
routine before any retained measurement. Nine batches per shape/slot give
1,728 observations per phase. All storage is preallocated; clocks and GC
counters accompany wall time. Every sample remains.

The prospective criteria are:

- Control phase: duplicate/copied and probe/copied within 1% in aggregate and
  2% in every worker, on every shape. Comparison retains the duplicate gate.
- Both phases: copied/actual within 3% in aggregate on every shape; every
  batch at least 20 ms; no measured GC; foreign CPU at most 2% of total
  capacity and steal at most 0.5%. Process groups remain below 1 GiB and
  180 seconds per worker.
- Comparison: both unpadded 30 and 128 improve at least 3% in aggregate
  against both copied and actual, with each worker's candidate/copied ratio
  at most 1.02. Other shapes allow at most 2% aggregate and 5% per-worker
  regression against copied.

These are empirical kernel criteria, not calibrated full-model confidence.
Failed control criteria stop the design. A successful kernel result still
requires actual AMD final-tier code inspection and separate product/model
qualification before any default decision.

## Reproduction

From the repository root, use fresh destinations and the exact core above:

```powershell
python -B tests/e5/softmax-reduction/generate.py --output artifacts/reduction-bench-new/kernels
dotnet build tests/e5/softmax-reduction-bench/Probe.csproj -c Release -o artifacts/reduction-bench-new/bin -p:FrozenCorePath=C:/Users/JoannesVermorel/code/Onnx/artifacts/softmax-zero-product-20260919/frozen/Lokad.Onnx.dll -p:KernelSourceDirectory=C:/Users/JoannesVermorel/code/Onnx/artifacts/reduction-bench-new/kernels --tl:off --nologo -v minimal
dotnet artifacts/reduction-bench-new/bin/Probe.dll artifacts/reduction-bench-new/local-check.json check 0
python -B -m unittest discover -s tests/e5/softmax-reduction-bench -p test_*.py -v
```

Save the successful test output as `audit-tests.log` in the artifact directory.
`prepare.py --artifact <directory>` freezes the source, binaries, generated
kernels, local check, test log and prospective plan. It pins the previous
closed workload hashes; the actual core must reproduce all of them. The
resulting `bundle.tar.gz` contains no model weights. Verify its digest before
extracting into a fresh VM directory. Do not run concurrently with another
VM inference lane.

On the VM, `python3 launch.py control` launches the control supervisor and
records its PID/start identity. Poll that same process until terminal, then
collect once with `python3 collect.py control`. Verify the archive and receipt
on both hosts; retain the archive as `control-results.tar.gz` and extract into
`collected-control`. The independent local audit is:

```powershell
python -B tests/e5/softmax-reduction-bench/audit.py --artifact artifacts/reduction-bench-new --phase control --output artifacts/reduction-bench-new/control-audit.json
```

A successful exit validates the evidence; the explicit `passed` field decides
the screen. Only when it is true, transfer that exact audit to the VM and run
`python3 launch.py compare <control-audit-sha256>`. Its gate verifies the prior
audit identity and absent supervisor/workers. Collect and audit comparison in
the same manner, using `compare-results.tar.gz`, `collected-compare` and
`compare-audit.json`. `export.py --artifact <directory> --output <new-json>`
reproduces the audit verdict and retains every raw batch in one report artifact.

Use `python -B` for auditing immutable extracted evidence to avoid creating
derived bytecode inside it. Every evidence writer refuses an existing output.
Never relaunch merely because a poll timed out. A separate code-capture worker
is excluded from performance and must use the same frozen binary.
