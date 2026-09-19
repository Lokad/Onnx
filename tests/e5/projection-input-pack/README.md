# E5 projection activation-packing experiment

The [September 19 result](results-20260919.md) rejects this candidate: all
correctness and duplicate-control checks pass, but the complete 30/128-row banks
regress. [Every measured sample](observations-20260919.json) is retained.

This standalone prototype copies each twelve-row activation group into
contiguous reduction order before running the existing 32-column AVX-512
kernel. Every candidate timing includes scratch rental, the complete activation
copy, destination clearing, computation, unchanged row tails and scratch return.
Constant weights remain prepared outside all steady-state timers. No production
kernel or default changes.

The original source supplies two identical control modes. The candidate keeps
the exact row partition, destination accumulation and FMA order. A portable
54-case packing proof checks every float bit, offsets, guards and input
preservation. Actual AMD computation adds 256 finite/exceptional geometry checks,
independent scalar-FMA coordinates, repeated accumulation and eight refusals per
worker. The local AVX2 host can verify packing and unsupported-compute refusal.

The fixed schedule has four fresh processes (orders 1, 2, 3, 4), followed by one
separate generated-code capture. Inference inherits CPU 2 before startup;
supervisor uses CPU 0. Runtime/GC settings are ordinary. Each worker is bounded
by 2 GiB sampled process-group RSS, 1,200 seconds and 256 MiB minimum available
system memory. All samples and failed results are retained.

Each worker measures twelve isolated shapes and four 72-matrix banks, with
seven samples per mode. The bank has 84,934,656 packed weight bytes. Its inputs
are synthetic and independent; these measurements are not full-model latency.
All 1,344 measured batches are retained. Seven-sample lists are preallocated,
and validation performs no intermediate JSON serialization.

The prospective screen uses within-worker median ratios and geometric means
over four workers. Both 30- and 128-row banks must improve at least 2% against
each control, with no worker regressing those cases by more than 2%. The 8/512
aggregate ratios must be at most 1.01. Duplicate controls must remain within
[0.98, 1.02] in aggregate and [0.95, 1.05] per worker. Failed controls mean
inconclusive; a failed gain criterion means no candidate nomination. A pass
requires a later product and complete-model protocol before any promotion.

From the repository root, using a fresh artifact directory:

```powershell
python -B -m unittest discover -s tests/e5/projection-input-pack -p test_*.py
python -B tests/e5/projection-input-pack/generate.py --output artifacts/projection-input-pack-next/source
dotnet build artifacts/projection-input-pack-next/source/Probe.csproj -c Release --tl:off --nologo -v minimal -p:FrozenCorePath=<absolute-pinned-core.dll> -o artifacts/projection-input-pack-next/bin
dotnet artifacts/projection-input-pack-next/bin/Probe.dll --host
python -B tests/e5/projection-input-pack/prepare.py --artifact artifacts/projection-input-pack-next
```

Preparation pins the existing qualified core, source transformations, binaries,
process helpers and prospective plan. It transfers no models. On Linux, extract
the length/SHA256-verified archive into a new directory, then invoke
`python3 -B <payload>/remote.py launch <payload>`. Poll its exact PID/start
identity. After every supervisor/worker/group is absent, `remote.py collect`
creates the archive and complete inventory. Verify both hashes locally, extract
only regular relative files into `collected`, and preserve returned metadata
as `download.json`.

```powershell
python -B tests/e5/projection-input-pack/audit.py --artifact artifacts/projection-input-pack-next --output artifacts/projection-input-pack-next/audit.json
```

Inspect the separately captured actual AMD assembly for input addressing,
vector spills and loop calls. Do not infer instructions from C# or treat a
source/isolated-kernel improvement as an ORT or whole-model result. Completed
writers must not be rerun; preserve failed attempts under their original scope.
