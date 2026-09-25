# Shared-model and e5 correctness of the packed final-row candidate

This adapter follows admitted Parakeet application closure b90aa8fc. It compares
qualified release Core f95a13c5 with candidate Core 49901366. The preceding
isolated M73 candidate 49c3a958 is the application baseline, not the release
baseline. Preparation binds both product lineages through their closed model
campaigns. Shared-model inference consumes Core alone; no Data DLL is loaded.

Use the exact existing Replay consumer and original native fixtures in four
fresh processes: selected-shared, selected-e5, candidate-shared, candidate-e5.
Each role covers 166 arrays and 5,000,814 values, including five e5 sequence/
padding cases, immutable inputs, held outputs, context reuse and memory policy.

Require every candidate float output to match the fresh current release byte
for byte. Independently compare both products against every original ORT array
with the unchanged 1e-4 scaled-error bound. This provides correctness evidence,
not a performance score. The active e5 repeatability diagnostic must finish,
be collected and audited before this adapter is staged on the VM.

The worker, resource protocol, numerical checks and independent auditor remain
byte-identical to the admitted M70 shared/e5 qualification. consumer_scope.py
checks that invariant. Only namespaces and prerequisite products change.
Retain 11 GiB available / 3 GiB tmpfs preflight, 8 GiB RSS, 900 seconds per
worker, 1 GiB minimum remaining memory/tmpfs and the original output bounds.
AMD CPU 2 computes; CPU 0 monitors. One VM workload runs at a time.

Use C:/Python313/python.exe -X utf8 -B followed by consumer_scope.py, then
run.py prepare, stage, launch, observe, collect and audit.py.
Freeze tools at preparation; redirect audit stdout outside the artifact directory.
Local namespace: artifacts/parakeet-packed-final-row-shared-amd-20260925.
VM namespace: /dev/shm/lokad-parakeet-packed-final-row-shared-20260925.
No inference, preparation or VM staging has run for this adapter yet.
