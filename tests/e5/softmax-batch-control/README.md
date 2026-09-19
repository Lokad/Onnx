# Longer softmax batches with duplicate controls

This is a distinct follow-up to the [failed zero-block screen](../softmax-zero-blocks/results-20260919.md).
The candidate arithmetic is unchanged. Fixed longer batches and conditioning
through the actual measurement routine address the earlier short intervals;
duplicate calls expose measurement variation before and during comparison.
Wall time remains the decision metric. Every sample is retained.
The [September 19 result](results-20260919.md) passes both phases.

The four timing slots are `actual`, `copied`, `duplicate` and `probe`.
`actual` invokes the frozen core; `copied` and `duplicate` reference the exact
same original-method delegate. In the control phase `probe` is another reference
to that delegate. In the comparison phase only `probe` changes to the adaptive
zero-block candidate. Every worker records its actual method identities.

Both phases use eight fresh Linux CPU-2 workers, four mode rotations and their
reversals. Six shapes are 8/30/padded128/128/512/padded512, with 12 times as many
rows as columns. Nine batches per mode use 32768/4096/256/256/32/32 calls.
Conditioning lasts at least one second per shape/mode in complete batches;
all conditioning ends before any retained measurement. Normal tiering and GC
remain enabled. All storage is preallocated, and hashing/JSON occur outside
the timed schedule.

Before comparison, both duplicate/copied and probe/copied controls must be
within 1% in aggregate and 2% in every worker, for every shape. During comparison,
the duplicate/copied control keeps those same bounds. All batches must last
at least 20 ms, with no measured GC, foreign CPU at most 2% of machine capacity
and steal at most 0.5% per worker. A failed control phase prevents comparison.

Candidate criteria preserve the earlier thresholds: at least 15% padded-128
gain in every worker against the copy and in aggregate against the actual core;
copied/actual control agreement within 3% on that shape; and unmasked regressions
at most 2% in aggregate and 5% in every worker. Padded-512 is a reported stress
case. These are empirical checks for this kernel experiment, not calibrated
whole-model confidence or permission to change defaults.

From the repository root, use a fresh artifact directory:

```powershell
python -X utf8 tests/e5/softmax-zero-blocks/generate.py --output artifacts/softmax-batch-next/kernels
$softmaxCore = (Resolve-Path artifacts/production-defaults-v2-20260919/frozen/Lokad.Onnx.dll).Path
$softmaxKernels = (Resolve-Path artifacts/softmax-batch-next/kernels).Path
dotnet build tests/e5/softmax-batch-control/Probe.csproj -c Release --tl:off --nologo -v minimal -p:FrozenCorePath=$softmaxCore -p:KernelSourceDirectory=$softmaxKernels -o artifacts/softmax-batch-next/bin
dotnet artifacts/softmax-batch-next/bin/Probe.dll artifacts/softmax-batch-next/local-check.json check 0
python -X utf8 tests/e5/softmax-batch-control/test_audit.py
python -X utf8 tests/e5/softmax-batch-control/test_gate.py
```

SDK 10.0.204 built the recorded probe. Its unchanged core SHA is
`7653c1686419d612e2624740908a44ffdee91b4239628a979bac44f5b6b863e9`.
The local check covers 2,503,712 exponential lanes, 918 tensors and four mask
refusals on an eight-lane vector host. Every AMD timing worker repeats them.

`prepare.py --artifact <fresh-artifact>` freezes the binary, generated kernels,
scripts and `.agent/m1-softmax-batch-controls-20260919.md`. It pins the earlier
workload input/mask/output hashes. Existing historical payloads and plans are
closed once reported; future experiments need a fresh prospective protocol.

After verifying a transferred payload and extracting it safely on Linux,
`python3 launch.py control` launches the control phase. The supervisor logs and
records its PID/start identity. Poll that same process until terminal. Only
then may `python3 collect.py control` create the archive and collection receipt.
Transfer both by verified hashes and extract into `collected-control` under the
local artifact, keeping the archive as `control-results.tar.gz`.

```powershell
python -X utf8 tests/e5/softmax-batch-control/audit.py --artifact artifacts/softmax-batch-next --phase control --output artifacts/softmax-batch-next/control-audit.json
```

The auditor writes explicit `passed` and individual criteria. A zero audit exit
code means the evidence was valid; it does not mean performance passed. If
`passed` is false, stop. If true, transfer the exact audit to the payload as
`control-audit.json`. `python3 launch.py compare <audit-sha256>` verifies the
receipt and the original terminal control identity before any comparison worker
can start. The five gate tests execute the actual supervisor's guard prefix,
including failed/mismatched audit, nonzero completion and a live supervisor,
without launching a worker.

Collect comparison the same way into `collected-compare` and
`compare-results.tar.gz`, then run:

```powershell
python -X utf8 tests/e5/softmax-batch-control/audit.py --artifact artifacts/softmax-batch-next --phase compare --output artifacts/softmax-batch-next/compare-audit.json
python -X utf8 tests/e5/softmax-batch-control/export.py --artifact artifacts/softmax-batch-next --output artifacts/softmax-batch-next/observations.json
```

The exporter includes only control if it failed; a passed control requires the
complete comparison result before export. It independently recomputes verdicts
from all raw samples. Do not rerun successful writers or overwrite evidence.
