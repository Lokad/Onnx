# M26 complete-LSTM screen

Compare current Core208371f6/Data b9358370 with isolated default-gate
Core5ee68015/Data1d8b2e40. The ordinary build, focused suites, complete
four-mode replay and actual generated-code review precede this screen.
Diagnostic closure9daee361 and independent reviewed helper ed4a98a2 preserve
the original interleaved JIT output and its explicitly proved extraction.

Reuse the byte-identical qualified M25 LstmScreen consumer, SHA256
1c4ac2412ecbb9aee483f5ae4314acbfa1a8636a40542e8355686d882f6dc60d.
Neither products nor consumer are rebuilt. Both products use the same
8192-byte input-row scratch. Two qualification processes precede four fresh
timing processes in selected, candidate, candidate, selected order.

Each timing process runs all twelve original complete captured LSTM graphs,
with equal case weight, one warmup and three measured passes. Width60 repeats
19 times and width256 repeats10 times per pass: 2352 timing clocks,588 warmups,
1764 measurements and48 graph preparations. Retain every sample; no exclusions
or unchanged retries. Qualification adds48 calls and24 graph preparations.

Reset plus Execute includes per-call weight panels/input buffer, both
projections, recurrence/gates and owned result allocation. Preparation and
numerical/ownership checks lie outside the timer. Require exact selected
output hashes, native scaled error<=1e-4, unchanged inputs/held outputs and
exact scratch on every call. Native ORT arrays are numerical references;
this screen does not measure ORT component or application latency.

Keep the existing ten repeatability controls: max/min<=1.10 aggregate and
<=1.20 per node. Keep five speed gates: candidate/selected<=.90 aggregate,
<=1.05 per node. A failed gate rejects the candidate for application work.
Admission still requires full product/package, model/native, long-meeting,
recovery and matched application qualification before integration or any
new published application/ORT ratio.

CPU2 workers/CPU0 monitor; no runtime/JIT flags; runtime10.0.8. Fixed timing
bounds:12 GiB available/3 GiB tmpfs preflight,8 GiB owned RSS,1 GiB live
available/tmpfs floor,900 seconds per job,1 GiB output and2 GiB artifacts.

From repository root use C:/Python313/python.exe -X utf8 -B with selftest.py,
then run.py prepare,stage,launch,observe,collect and audit.py in this folder.
Fresh artifact: artifacts/pyannote-lstm-gates-screen-amd-20260923.
Fresh VM directory: /dev/shm/lokad-pyannote-lstm-gates-screen-20260923.
Refuse existing namespaces; freeze tools before workers; collect after every
recorded PID/birth is terminal. Preserve failures explicitly. Do not observe
again after audit closes the local ledger.
