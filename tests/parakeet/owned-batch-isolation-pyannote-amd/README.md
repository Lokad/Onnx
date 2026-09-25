# Complete Pyannote correctness after the Parakeet dispatch relocation

Compare release Core f95a13c5 / Data a893952f with candidate Core e07a4518 /
Data 01e9e784. Preparation requires admitted direct-release Parakeet performance
and completed shared-model correctness for these actual products. Run serially
after those campaigns close, using existing assets only.

Reuse the two already verified GraphQualification consumers. The candidate's
Data identity matches the earlier literal-only consumer, 8c21f206; the selected
consumer is 94210bcb. Retain the original compiled comparison of all 96 methods,
with 95 unchanged and only the Main Data-identity literal different. Verify
the exact consumer binaries and dependency files against their original closed
receipts. No product, consumer, restore or inventory process is rebuilt or rerun.

Two fresh inference processes execute all 18 complete graph arrays / 2,917,107
values and 16 public calls per product. Keep the original native 1e-4 bounds,
byte-identical release tensors and exact complete public results, including
centroids, speaker assignments, intervals, status, input immutability and held
outputs. The original numerical validators and seven semantic tests are exact.
No performance score is produced here; earlier model outputs are not reused as
the candidate's new correctness proof.

consumer_scope.py verifies the narrow adapter changes: remove the three build
and inventory jobs, verify their retained consumer proof, update product labels
and namespaces, and collect the actual runtime binaries. Numerical checks,
inference commands, resource monitoring and limits remain unchanged.

Limits remain 11 GiB available RAM / 3 GiB tmpfs before execution, 8 GiB RSS,
900 seconds per job, 1 GiB remaining RAM/tmpfs and the original output bounds.
AMD CPU 2 computes; CPU 0 monitors. Keep one VM workload at a time.

Use C:/Python313/python.exe -X utf8 -B with consumer_scope.py, test_semantics.py,
then run.py prepare, stage, launch, observe while live, collect once terminal,
and audit.py once. Tools freeze at preparation; keep audit stdout outside the
artifact directory. Preserve every result and never rerun a closed campaign.

Local: artifacts/parakeet-owned-batch-isolation-pyannote-amd-20260925.
Remote: /dev/shm/lokad-parakeet-owned-batch-isolation-pyannote-20260925.
