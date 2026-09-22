# Fresh AMD comparison of complete Parakeet transcription

Compare selected production Core `1279b4b6` / Data `4e602d9f`, the qualified
Parakeet arithmetic composition Core `abbf5e98` / Data `eb452663`, and Microsoft
ONNX Runtime 1.29.0. Both managed roles retain the selected Pyannote convolution
and 256 MiB encoder packing cap. Pyannote remains first, Parakeet second and
Whisper deferred. Original failed trials remain closed and no samples are reused.
The protocol's internal candidate role is named `portable`.

Before timing, require normal Linux builds matching all 3,114 Core / 697 Data
methods and public declarations, at least 3,313 backend / 343 tensor passes,
and executed AVX-512 tests. Run 400 convolution cases in both modes; all original
416 raw / 416 geometry / 48 dynamic / four actual-operand / four disabled checks;
and forty AMD prepared-dispatch precedence cases. The new consumer qualification
is closed at `967f94c89c29852c9b1db93fdf89567acab2c8c1abe0bfc51232576c702e89d4`.

Fresh graph/native checks cover 36 Pyannote arrays / 32 public requests and
1,568 Parakeet arrays across both managed roles. Require Pyannote outputs exact
to selected production and every original native bound. Additionally execute
all twenty public Parakeet clips for each managed role and ORT (60 requests).
Retain the separate four-request Pyannote native conformance. Both 600-second
Pyannote meetings and 30-second recovery precede performance admission.

Six fresh timing processes run production, candidate, ORT, ORT, candidate,
production. Each performs one warmup and three measured passes over the same
20 clips / 213.265 audio seconds: 480 requests, 120 warmups, 360 measurements.
Sum the twenty per-clip means within each process, then average both process
corpus means equally. All statistics and gates derive from integer clocks.
Require each role's corpus process max/min <=1.10 and every clip <=1.20;
candidate corpus/production <=0.95 and every clip <=1.05. All 63 controls and
21 speed gates are mandatory. Full application parity <=1.05 to ORT is separate.
No sample deletion, gate relaxation or unchanged retry follows a failed result.

CPU2 before CLR/native startup; monitor CPU0. SDK10.0.204 / .NET10.0.8; native
ORT uses one intra/inter-op thread, sequential execution, full optimization,
no spinning. No JIT/GC overrides or profiling in timing. Preserve four-hour
campaign / one-hour worker ceilings, 12 GiB owned RSS, 1 GiB available/tmpfs,
2 GiB artifacts and 3 GiB tmpfs preflight. Public Parakeet workers require
14 GiB available; other workers require 12 GiB. Retain bounded preflight waits.
After verified transfer, remove duplicate archives; after all builds/tests,
remove only inventoried, unpinned build caches. Preserve every input/source,
built binary and result for final verification and collection.

From repository root use `C:/Python313/python.exe -X utf8 -B prepare.py` with
this directory's script path. Preparation freezes all inputs and runs selftests.
After successful preparation run `finish.py` once. Artifacts are
`artifacts/parakeet-single-panel-amd-payload-20260922` and
`artifacts/parakeet-single-panel-amd-execution-20260922`; remote is
`/dev/shm/lokad-parakeet-single-panel-20260922`. Existing paths are refused.
The VM is already authorized. Collection waits for actual terminal owners and
independent audit recomputes every gate. Root arithmetic integration requires
an admitted result followed by normal root/package qualification.
