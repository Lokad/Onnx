# Positional slice conversion attribution

This single-use campaign compares current Core f95a13c5 with the full-model
qualified slice-conversion Core 49c3a958. It reuses the certified wall observer
and consumer unchanged; it builds nothing. Both processes execute all20 clips,
one warmup and three measured passes. CPU2 executes and CPU0 monitors.

The prediction was published before timing in
`../slice-dense-conversion-results/groups-20260925.json`: the complete34-node
positional group and every one of its24 MatMul kernels improve. Shared
preparation is counted once. Every other node and phase remains in the report.
All original public checks, interval reconciliation, foreign CPU and resource
limits remain. Profile clocks include observer overhead and cannot establish
application improvement or a new ORT ratio.

From the repository root, use Python3 with `-B` to run this directory's `run.py`
actions `prepare`, `stage`, `launch`, then `observe` only while its owner is
live. Once terminal, `collect` and `audit.py` each run once. All inputs and tools
are immutable after preparation. Preserve any failure and diagnose it from
retained evidence; never repeat completed workload legs.
