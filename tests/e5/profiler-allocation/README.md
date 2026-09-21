# Disabled-profiler allocation candidate

This isolated candidate moves the captured node callback into a detailed-only
profiling branch. The production source tree and frozen VM campaigns stay intact.

From the root run `prepare.py`, `run.py`, then the audit with
`C:/Python313/python.exe -X utf8 -B`. Preparation archives source `d875d99`, builds
both cores, and freezes identical consumers. Two sequential Windows CPU-2 workers
cover five e5 cases, two policies and disabled/detailed/wall profiles, each with
16 warmup and 16 observed executions. Allocation counts are the primary result;
this lane makes no timing or AMD performance claim.

Compiled instructions, full first/last outputs, every output hash/native check,
profile descriptions and resources are retained under
`artifacts/e5-profiler-allocation-20260921`. Preparation and execution refuse
overwrites. Promotion requires separate complete-model timing and affected-model
qualification as specified in PLAN.md.
