# One LSTM layout experiment

The ORT-guided hypothesis is in
[the exact-case review](../decoder-lstm-current-review/next-decision-20260927.md).
The [contract diagnosis](../decoder-lstm-layout-results/contracts-20260927.md)
resolves the existing unsupported explicit-FMA tests while retaining their
original failure. Only the frozen `ad97b4ad` candidate and qualified `0d224bcf`
control are measured. No product rebuild or alternative layout is included.

Reuse the existing complete-LSTM `Timing.cs` consumer. Its only adaptations allow
two fallback role names and correct residency checks: both prepared products now
retain 13,107,200 bytes per graph. The fallback roles set graph preparation capacity
to zero and require zero retained weights. All 380 captured calls, original order,
input/output validation, held-output and immutability checks, allocation counters
and timer boundaries remain. All three outputs must match the original capture
exactly. Setup/preparation is recorded separately. Every inference clock includes
reset, validation, allocation and complete LSTM execution.

Before measurements, the earlier projection-only screen becomes a complete-call
screen with the same 10% gain requirement. This wider boundary tests whether the
single change saves meaningful LSTM time. A fixed unprepared-path control verifies
path specificity. It is not another candidate. No projection-only latency claim
will be inferred from complete-call clocks.

For each prepared/fallback path and normal/AVX512-disabled mode, run
current/candidate/candidate/current. Each process retains five complete warmup
and five measured passes. There are 16 processes, 60,800 clocks (30,400 warmup and
30,400 measured), 182,400 exact output checks, 168 repeatability controls and 28
performance gates. Require every within/between-process ratio <=1.10, prepared
corpus candidate/current <=0.90, no prepared case >5% slower and every fallback
corpus/case ratio <=1.05. Retain all clocks and failures; never trim or retry an
unchanged failed screen.

Reuse the original serial worker, PID/birth ownership, CPU 2 worker/CPU 0 monitor,
foreign CPU <=1%, SDK 10.0.204/runtime 10.0.8 and resource bounds. Stage only with
8 GiB available RAM and 2 GiB tmpfs, stop above 4 GiB worker RSS or 1 GiB artifacts.
Root source and BENCHMARK.md remain unchanged until complete release qualification.

From repository root, execute `C:/Python313/python.exe -X utf8 -B
tests/parakeet/decoder-lstm-layout-timing/run.py` with actions `prepare`, `stage`,
`launch`, then `observe`. Each mutation is once only; do not replay completed
namespaces. After all owners are terminal, `collect` once and execute `audit.py`
once. Failed controls require diagnosis before an application trial or code change.
