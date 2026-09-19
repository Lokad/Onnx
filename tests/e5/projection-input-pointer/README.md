# Packed-input pointer proof and complete-cost comparison

The preceding [index-based experiment](../projection-input-pack/results-20260919.md)
is closed and rejected. This separate prototype replaces twelve computed
`input[j * 12 + row]` indexes with literal `input[row]` reads, advancing the
pointer by twelve floats per step. Packing, row partition, destination
accumulation, FMA order and all tails stay identical.

Four modes distinguish the original, its identical duplicate, the rejected
indexing implementation and the new pointer implementation. Candidate timings
include scratch rent, complete activation copy, destination clear, compute,
tails and scratch return on every matrix call. Constant B weights alone are
prepared outside steady-state timers. No product code/default changes.

The first AMD worker performs complete correctness and captures actual FullOpts
code, then exits without timing. The code auditor requires 24 FMAs, twelve
broadcasts from one base at literal byte offsets 0 through 44, a 48-byte pointer
advance, no vector spills or loop calls, and at most two sign extensions and
two LEAs in the hot loop. Timing preparation requires this passed proof and its
closed receipt. It uses the exact qualified binaries without rebuilding.

The subsequent fixed schedule has four fresh processes, orders 1/2/3/4. Each
checks 256 finite/exceptional matrix geometries, 54 packing-layout cases and
24 invalid-shape calls across the three kernel implementations. Full bits,
scalar-FMA coordinates, nonzero/repeated accumulation, input/weight preservation
and guards are checked. The local AVX2 host checks packing and compute refusal.

Timing covers twelve isolated projection shapes and four 72-matrix banks,
retaining seven samples per mode: 1,792 measured batches. The bank contains
84,934,656 packed bytes with independent synthetic inputs. A prototype pass
cannot establish full-model or ORT performance. CPU 2 is inherited before
startup, supervisor CPU 0; normal runtime/GC and per-worker limits of 2 GiB RSS,
1,200 seconds and at least 256 MiB available memory apply. Only proof capture
sets JIT output flags.

The prospective nomination rule requires at least 2% aggregate improvement
at 30 and 128 rows against each original control, with no worker regressing
those banks by more than 2%. At 8/512 rows, aggregate ratios must be at most
1.01. Original duplicate ratios must stay within [0.98, 1.02] aggregate and
[0.95, 1.05] per worker. Ratios use seven-sample medians and geometric means
across workers. Keep every sample; failed controls are inconclusive, and failed
gain/code requirements reject nomination. Beating only the old indexing route
is insufficient. No threshold fitting or automatic rerun follows failure.

Use fresh destinations from the repository root:

```powershell
python -B -m unittest discover -s tests/e5/projection-input-pointer -p test_*.py
python -B tests/e5/projection-input-pointer/generate.py --output artifacts/pointer-proof-next/source
dotnet build artifacts/pointer-proof-next/source/Probe.csproj -c Release --tl:off --nologo -v minimal -p:FrozenCorePath=<absolute-pinned-core.dll> -o artifacts/pointer-proof-next/bin
dotnet artifacts/pointer-proof-next/bin/Probe.dll --host
python -B tests/e5/projection-input-pointer/prepare.py --phase proof --artifact artifacts/pointer-proof-next
```

Transfer the verified archive into a new Linux directory, then use its
`remote.py launch <payload>`. After actual termination, `remote.py collect`
verifies all process identities/groups and inventories the results. Verify
archive length/SHA256 locally and extract regular relative files into
`collected`; preserve returned metadata as `download.json`.

```powershell
python -B tests/e5/projection-input-pointer/audit.py --artifact artifacts/pointer-proof-next --output artifacts/pointer-proof-next/audit.json
```

Close the passed proof with a receipt binding the audit, binaries and full file
inventory before preparing timing with `prepare.py --phase timing --artifact
artifacts/pointer-timing-next --proof-artifact artifacts/pointer-proof-next`.
Collect and audit timing through the same commands with its own destination.
Completed writers refuse existing outputs. Preserve the closed predecessor and
all first failures; neither an observation timeout nor failed screen authorizes
an unchanged restart.
