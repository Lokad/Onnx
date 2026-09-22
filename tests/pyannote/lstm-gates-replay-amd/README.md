# Exact-product default-gate LSTM replay

Reuse the unchanged qualified M25 ordinary consumer
1a057180ce16082fe5bea2fc50ab4f490b1e8cf75a4c743b715dc491276b4c09.
No product or consumer build occurs. Fresh current/candidate runtimes retain
their complete original dependencies. The eight processes cover each product
with AVX512 disabled, ordinary AVX512, hardware intrinsics disabled and explicit
SIMD-only. Both products include M22 and use identical input-row scratch.

Every original captured graph, attribute, optional slot and input remains.
Each process checks24calls/72outputs/3631104values against exact AMD selected
bits and native ORT scaled error<=1e-4, readonly operands, held owned outputs
and exact scratch. Totals192calls/576outputs/29048832values. Reconstruct
reference provenance independently after collecting terminal workers.

Before workers freeze10GiB available/3GiB tmpfs preflight,8GiB ownedRSS,
1GiB live available/tmpfs floor,900seconds/job,1GiB output,2GiB artifacts.
CPU2 workers/CPU0 monitor, runtime10.0.8 and original qualified toolchain.
Only declared hardware flags are permitted. This is unscored correctness;
no performance or integration claim follows from success.

From repository root use C:/Python313/python.exe -X utf8 -B with run.py
prepare,stage,launch,observe,collect then audit.py. Namespace
artifacts/pyannote-lstm-gates-replay-amd-20260923 and
/dev/shm/lokad-pyannote-lstm-gates-replay-20260923. Refuse existing destinations.
Use observe only before sealing; later status reads must not append to sealed
evidence. Preserve original failures with explicit successors.
