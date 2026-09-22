# M25 complete-LSTM screen

Compare the exact current Core208371f6/Data b9358370 and isolated wide
Core63d59bad/Data90938e53. All numerical and generated-code checks precede
this screen. The diagnostic closure is reconciled-closed.json,
55dfae31762b55fd2c124404506a270662571eaf3aef138bf534e160c0b16f0b.
No product rebuild occurs. The retained M22 Screen.cs changes one expression
only: both current products now use the same 8192-byte input-row scratch.
Compare every compiled consumer method against the original qualified driver;
only Main may differ and the public surface must remain equal.

SDK identity, offline consumer restore/build and compiled inventory precede
two qualification workers. Four fresh timing processes then run selected,
candidate, candidate, selected. Each runs all twelve original captured
complete LSTM graphs with equal case weight, one warmup and three measured
passes. Width60 repeats19 times and width256 repeats10 times per pass:
2352 timing clocks, 588 warmups, 1764 measurements and 48 graph preparations.
Retain every sample without exclusions or unchanged retries.

Reset plus Execute includes per-call weight panels/input buffer, both
projections, recurrence/gates and owned result allocation. Preparation and
all numerical/ownership checks lie outside the timer. Require exact selected
output hashes, native scaled error<=1e-4, unchanged inputs/held outputs and
exact scratch on every call. ORT arrays are numerical references; this screen
does not measure native component or application latency.

Freeze the existing ten repeatability controls: max/min<=1.10 aggregate and
<=1.20 per node. Freeze five speed gates: candidate/selected<=.90 aggregate,
<=1.05 per node. A failed gate rejects the candidate for application work.
An admitted component still needs full model/package/application qualification
before integration or any new published application/ORT ratio.

CPU2 workers/CPU0 monitor; no runtime/JIT flags; SDK10.0.204/runtime10.0.8.
Keep the timing resource bounds: 12 GiB available/3 GiB tmpfs preflight,
8 GiB owned RSS, 1 GiB live available/tmpfs floor, 900 seconds per job,
1 GiB output, 2 GiB artifacts. Build only on the authorized VM.

From repository root, use C:/Python313/python.exe -X utf8 -B with selftest.py,
then run.py prepare,stage,launch,observe,collect and audit.py in this folder.
Fresh artifact: artifacts/pyannote-lstm-wide-screen-amd-20260923.
Fresh VM directory: /dev/shm/lokad-pyannote-lstm-wide-screen-20260923.
Refuse existing namespaces. Freeze tools before workers; collect only after
every recorded PID/birth is terminal. Preserve any failed attempt explicitly.
