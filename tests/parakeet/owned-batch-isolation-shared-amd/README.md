# Qualify shared-model contracts after the Parakeet dispatch relocation

Compare the actual release Core f95a13c5 with relocation Core e07a4518.
Preparation requires the admitted direct-release Parakeet application comparison
and complete model correctness for the exact candidate. No Data assembly is
loaded by these shared-model checks. Do not prepare or stage while the preceding
application campaign is running.

Reuse the original Replay consumer and native fixtures in four fresh processes:
selected-shared, selected-e5, candidate-shared and candidate-e5. Each product
covers 166 arrays / 5,000,814 values, including five e5 shapes and padding cases,
immutable inputs, held outputs, execution-context reuse and memory policy.
Require candidate outputs byte-identical to the fresh release outputs, and both
products within the original 1e-4 scaled-error bound against every ORT fixture.
This is correctness qualification, with no performance score or new variant.

The numerical checks, worker, resource protocol and independent auditor are
unchanged. consumer_scope.py verifies their exact original bytes. Only product
prerequisites and campaign namespaces are rebound from packed-final-row-shared-amd.
No product or consumer is rebuilt. Use the retained local and VM assets.

Limits remain 11 GiB available RAM / 3 GiB tmpfs before execution, 8 GiB RSS,
900 seconds per worker, 1 GiB remaining RAM/tmpfs and the original output limits.
AMD CPU 2 computes and CPU 0 monitors; run only one VM workload at a time.

Use C:/Python313/python.exe -X utf8 -B with consumer_scope.py, then run.py prepare,
stage, launch, observe while live, collect once terminal, and audit.py once.
Tools freeze at preparation. Keep audit stdout outside the artifact directory.
Preserve every result and do not rerun a closed campaign.

Local: artifacts/parakeet-owned-batch-isolation-shared-amd-20260925.
Remote: /dev/shm/lokad-parakeet-owned-batch-isolation-shared-20260925.
