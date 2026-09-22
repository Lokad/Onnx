# Actual Pyannote layers through ordinary graph execution

This numerical qualification uses the M17 normal Core binary after its compiled
method review and31normal/31hardware-disabled focused checks. It creates108small
graphs from the retained actual layer operands, with ordinary Conv/ConvRelu and
Add/AddRelu nodes. All32eligible and4fallback calls per crop remain included.

The original fixture loader, direct public baseline, full119,823,360-value checks,
native tolerance, repeats, held outputs and read-only checks are preserved.
Only candidate dispatch, identity and additional observations change. The original
component helper remains solely for the fixture loader's reference weight packing;
the candidate always executes the normal graph/provider path. Exact requested
scratch bytes prove eligible dispatch, alongside bounded prepared residency.

Every graph runs twice. The108independent qualification graphs retain three
copies of the21,086,208-byte weight census; this is a test arrangement, not a
claim about full-model memory usage. Whole-model/public application qualification
and AMD target execution remain separate requirements. No timing is scored.

From root using C:/Python313/python.exe -X utf8 -B, run build.py then audit.py.
Artifacts: artifacts/pyannote-blocked-spatial-layer-graphs-20260922. Resource bounds
remain8GiBbuild/12GiBnumerical available preflight,8GiBRSS,1GiBminimum available,
20GiBfree disk,900seconds per worker, CPU2 before runtime and monitorCPU0.
