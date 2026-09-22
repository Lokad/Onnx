# Complete current/candidate LSTM numerical replay

Both products include M22 and therefore require exactly8192bytes of input-row
scratch when hardware panels are admitted. The retained replay consumer changes
only Main: five declared source replacements add a SIMD-only mode, require its
actual hardware availability and correct the selected role's scratch expectation.
Every original tensor, graph, optional slot, output, native bound, repeated-call,
input immutability and held-output check remains. Compare every compiled method
against the original consumer: only Main may differ; no additions/removals or
public declaration changes. Build only the consumer on AMD, not either product.

Four preparation jobs (SDK identity, offline restore, build, inventory) precede
eight fresh processes. Each selected/candidate product runs AVX512-disabled,
ordinary AVX512, hardware-disabled and explicit SIMD-only execution. Each replays
all12original captured graphs twice, all3outputs and3,631,104values:192calls,
576outputs and29,048,832values overall. Exact AMD selected bits, native scaled
error<=1e-4, full input identity and owned outputs are mandatory. SIMD-only
numerical agreement does not alone establish routing; separate JIT evidence
must verify dispatch before timing.

From repository root use C:/Python313/python.exe -X utf8 -B with test_checks.py,
then run.py prepare,stage,launch,observe,collect and audit.py. New namespaces:
artifacts/pyannote-lstm-wide-replay-amd-20260923 and
/dev/shm/lokad-pyannote-lstm-wide-replay-20260923. Refuse existing destinations.
The local auditor independently reconstructs all AMD/native reference arrays
and provenance using the original selected-platform evidence.

Before any worker, freeze10GiB available/3GiB tmpfs preflight,8GiB owned RSS,
1GiB live available/tmpfs floor,900seconds per worker,1GiB output and2GiB
artifacts. These are unscored numerical jobs with the same bounds as the focused
consumer/tests, declared before execution. Worker CPU2/monitor CPU0; SDK10.0.204,
runtime10.0.8, original offline toolchain, no extra runtime flags. No completed
trial's limits or scored gates change. Collect all files only after terminal
owners; no performance or integration claim follows from numerical success.
