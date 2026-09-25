# Long-input e5 frequency observation

Two identical M73 candidate processes retain the original 600 warmup and 180
measured calls, consumer, inputs, output checks, runtime flags and EventPipe
collection. No product is built and no release score is produced.

The exact qualified `../e5-frequency-probe-v3/counter.py` observes CPU 2 through
installed perf. Worker threads stay on CPU 2; the observer and EventPipe stay on
CPU 0. Only perf's owned process tree may temporarily select CPU 2 to read its
counters. Every PID/birth, affinity, resource sample and raw interval is retained.

Use `C:/Python313/python.exe -X utf8 -B` with `run.py prepare`, `stage`, `launch`,
`observe`, `collect`, then `audit.py`. Each mutating step refuses overwrite.
No launch is allowed without the retained successful clock proof. Frozen source
and all prior failed benchmark controls remain immutable.

The hypothesis predicts slower fixed 60-call blocks accompanying lower APERF /
MPERF. Every block keeps all clocks. Only whole counter intervals inside both
clock bounds contribute, with at least 85% duration coverage. All other intervals
are retained with a reason. Ratios describe VM counters, not calibrated host MHz;
association cannot establish the cause of a previous run.

The initial spare-memory reservation is 128 MiB for these two captures: twice
the largest retained trace plus compressed export, copied inputs, and 32 MiB
for other output must fit it before freezing. Immutable binaries are hard links.
The original 11 GiB per-worker preflight and all runtime hard limits are unchanged.
