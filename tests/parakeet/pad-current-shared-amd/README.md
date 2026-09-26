# Shared-model correctness for the current padding dispatcher

Compare qualified Core f3992f40 with copy dispatcher Core a74acb17. Prepare only
after the current complete Parakeet application comparison closes and passes.
No Data assembly is loaded in this lane. These are correctness checks with no
performance score, new optimization, consumer build or model download.

Reuse the original Replay a50d3e96 and all native fixtures. The prior shared-model
closure qualified that exact consumer against e07a4518. The closed current-model
compatibility proof binds e07a4518 to qualified root f3992f40 with unchanged
method bodies/flags and all old public bindings, then to the four reviewed Pad
call changes and private copy helper. Both new products must still execute all
checks afresh; binary compatibility is not their behavioral qualification.

Four fresh processes run selected-shared, selected-e5, candidate-shared and
candidate-e5. Each product covers 166 arrays / 5,000,814 values, including five
e5 shapes/padding cases, input immutability, held outputs, execution-context reuse
and memory policy. Require candidate tensors byte-identical to fresh current-root
outputs and the original 1e-4 scaled bound against every ORT fixture.

The numerical checks, worker, protocol and auditor are byte-identical to the
previous shared lane. Staging changes only source namespaces. Preparation adds
the exact compatibility and previous shared-consumer proofs; consumer_scope checks
preserve all fixture and asset handling. All six failed component repeatability
controls remain recorded in the compatibility proof; they are not rescored here.

Preserve 11GiB available RAM / 3GiB tmpfs preflight, 8GiB RSS, 900 seconds/job,
1GiB remaining RAM/tmpfs and original output bounds. CPU2 computes, CPU0 monitors;
run one VM workload at a time. Use existing models and the offline environment.

With C:/Python313/python.exe -X utf8 -B run consumer_scope.py and test_prerequisites.py.
After application admission, use run.py prepare, stage and launch once each;
observe the same owner, then collect and audit.py once all owners are terminal.
Keep audit stdout outside the artifact directory. Tools freeze at preparation.

Local: artifacts/parakeet-pad-current-shared-amd-20260926.
Remote: /dev/shm/lokad-parakeet-pad-current-shared-20260926.
Pyannote correctness/performance, graph regressions and actual root/package
qualification remain required before product promotion.
