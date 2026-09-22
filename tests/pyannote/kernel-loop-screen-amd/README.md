# M23 fixed complete-call screen

Compare the current M22 product (Core208371f6) with isolated M23
(Core29620671), after complete AMD numerical and generated-code qualification.
Reuse the exact qualified SpatialWeightScreen driver, scorer and geometry
manifest from the prior spatial-weight screen. The same qualified LayerGraphs
caller serves both roles; the driver invokes Fixtures/Candidate/GraphCalls
helpers directly and never invokes ModelProbe.Main. No consumer logic, product
or driver is rebuilt here.

From the repository root, use Python 3.13 `-X utf8 -B`: run `test_score.py`,
then `run.py prepare`, `stage`, `launch`, `observe`, `collect`, and `audit.py`.
Fresh directories are `artifacts/pyannote-kernel-loop-screen-amd-20260922` and
`/dev/shm/lokad-pyannote-kernel-loop-screen-20260922`. Preserve failed runs;
no unchanged timing retry is authorized by this protocol.

Four fresh AMD processes run current, candidate, candidate, current. Original
geometry fixes 1,074 iterations per pass, one warmup and three measured passes:
17,184 call clocks and 512 separate graph-creation/preparation clocks. All 108
calls, including fallback calls and original form multiplicities, contribute.
No calibration, trimming, profiling, ISA or tiering overrides are allowed.

Timing includes caller assertions/dispatch recording, finite scans, scratch,
conversions, kernel and graph epilogues. Hashing and journal writes are outside.
Numerical hashes, ownership, preparation and original resource checks remain.
CPU2 is set before worker startup; the monitor uses CPU0. Frozen limits are in
`protocol.py`. No other workload may overlap the timing campaign.

Exact integer-clock fractions determine all 32 repeatability controls and
12 speed gates: process max/min ≤1.10 aggregate / ≤1.20 each form;
candidate/current ≤0.90 aggregate / ≤1.05 every eligible form. These are fixed
before execution. Passing admits full product/application qualification, not
root integration or application parity. A rejection retains all evidence and
leaves the selected product unchanged. Full application/ORT target remains
≤1.05; Pyannote first, Parakeet second, Whisper deferred.
