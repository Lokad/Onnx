# Shared-model and e5 correctness for rational sigmoid

Compare qualified actual-root Core `8bb22038` with the fixed rational candidate
`946ddfb6`. The completed Parakeet application comparison admits its 3.248966%
gain. This lane checks broader correctness; it does not admit a release or claim
that the rejected isolated operator screen now passes. Its 13 failed controls,
four fallback regressions and unresolved double latency remain documented.

Reuse original Replay `a50d3e96`, all native fixtures and existing models. No Data
assembly is loaded, and no consumer/product build or download is required. The
closed model compatibility proof reconciles 3,979 existing methods from the
previously measured padding product through actual root to this candidate. Only
Sigmoid changes and one private helper is added; old public bindings and method
flags remain intact. Both products must execute the behavioral checks afresh.

Four processes run selected-shared, selected-e5, candidate-shared, candidate-e5.
Each covers 166 arrays / 5,000,814 values, including all five e5 shapes/padding
cases, immutable inputs, held outputs, context reuse and memory policy.
The original finite-output, shape, ownership and ORT scaled-error bound remain:
`abs(actual-reference)/max(1,abs(reference)) <= 1e-4` for every value.
The padding lane's extra candidate/current byte-equality condition becomes the
same scaled numerical bound, declared before execution for the new arithmetic.
Every candidate/current maximum and byte-equality result is recorded separately.
Native validation is unchanged; numerical agreement is not a performance claim.

Protocol, worker and auditor are byte-identical to the closed padding lane.
`consumer_scope.py` verifies every checker adaptation and preserves preparation's
complete fixture/asset handling. `prerequisites.py` binds the exact closed model,
previous shared consumer and admitted application products, with all failures retained.

Resource limits remain 11 GiB available RAM / 3 GiB tmpfs before work, 8 GiB owned
RSS, 900 seconds per job, and 1 GiB remaining RAM/tmpfs. CPU 2 computes and CPU 0
monitors. Only one VM workload may run at a time. Tools freeze at preparation.

From the repository root, prefix each command with
`C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/rational-sigmoid-shared-amd/consumer_scope.py
    tests/parakeet/rational-sigmoid-shared-amd/test_prerequisites.py
    tests/parakeet/rational-sigmoid-shared-amd/run.py prepare
    tests/parakeet/rational-sigmoid-shared-amd/run.py stage
    tests/parakeet/rational-sigmoid-shared-amd/run.py launch
    tests/parakeet/rational-sigmoid-shared-amd/run.py observe

Observe the same owner until terminal, then collect and run `audit.py` once,
keeping audit stdout outside the artifact. Do not retry a completed campaign.
Local artifact: `artifacts/parakeet-rational-sigmoid-shared-amd-20260927`.
VM: `/dev/shm/lokad-parakeet-rational-sigmoid-shared-20260927`.
Pyannote, graph regressions and actual-root/package qualification still follow.
