# Prove counter timestamps without running a model

This proof precedes the two-process long-input e5 frequency observation. It uses
installed perf 6.17.13 with 1000 ms intervals on CPU 2, controlled through private
FIFOs. A timestamp before each enable/disable request and after its complete
forced interval output bounds perf's relative CLOCK_MONOTONIC epoch. Both bounds
must intersect and have at most 20 ms uncertainty. Every raw interval remains.

The same proof brackets 1,000 calls to the installed .NET 10.0.8
SystemNative_GetTimestamp function between Python monotonic_ns readings. A small
2.3-second arithmetic workload must contain an entire aligned frequency interval
with positive APERF/MPERF/TSC values. No model, product rebuild, scoring change or
runtime/frequency setting change is involved.

Eight directed parser/alignment tests must pass before launch. The proof is
bounded to 30 seconds, 512 MiB owned RSS and 64 MiB output. A bounded helper
terminates root perf through a stop file; recorded owners must all be terminal
before collection. perf may briefly bind to CPU 2 to read its counters; retain
that observed affinity. The workload is restricted to CPU 2 and monitor to CPU 0.

Use C:/Python313/python.exe -X utf8 -B with test_counter.py, then run.py launch,
observe and collect. All tools freeze at launch. Audit separately after terminal
collection. Local namespace: artifacts/e5-frequency-proof-amd-20260925.
VM namespace: /dev/shm/lokad-e5-frequency-proof-20260925.
The executable plan is .agent/m79-e5-frequency-observation-20260925.md.
