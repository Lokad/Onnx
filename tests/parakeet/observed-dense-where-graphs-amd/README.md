# Full graph regression comparison for the observed-mask candidate

Run after the M70 Parakeet application is admitted and fresh shared/e5 and
Pyannote correctness checks pass. Compare the exact current and candidate
Core binaries with fresh ORT processes on all eight existing cases: five e5
sequence/padding inputs, DINOv3, ResNet50 and GPT-2. Reuse canonical model files,
original fixtures, numerical bounds, output checks and all scoring gates.

Use the previously qualified fixed warmup for each case. Thirty-token e5 uses
the existing consumer 0b228b2d with 1,200 warmups; the other seven use d827e3b9
with 600 warmups. Both retain 180 measurements. Their native counterparts are
the exact corresponding original scripts. Reuse both compiled consumers and
their complete method/flag reviews; no new consumer build or runtime override.
This reproduces the protocols in the current qualified release, prospectively.

There are 72 fresh processes: three numerical processes per case, then six
timing processes in current, candidate, ORT, ORT, candidate, current order.
Retain all 41,112 calls, 8,640 measurements and 72 setups. Each engine must pass
its original process repeatability bound and each candidate case must remain
within 5% of the current release. Retain every clock, result, input/held-output
check and original 1e-4 native bound. No trimming or unchanged timing retry.

The two statistics implementations remain byte-identical to their qualified
versions. A wrapper routes only by the already validated case identity.
The worker changes only which existing consumer/native script serves 30-token
e5. The request checker changes only the corresponding identity and fixed
counts. consumer_scope.py verifies that scope, and both original statistics
test suites and the original inventory tests remain.

Bounds stay 11 GiB available / 3 GiB tmpfs before each process, 8 GiB RSS,
1 GiB remaining memory/tmpfs, 1 GiB output/job, 2 GiB stage, 900 seconds/job and
four hours total. CPU 2 computes; CPU 0 monitors. No concurrent VM workload.
Only prepare after all prerequisites pass. Use C:/Python313/python.exe -X utf8 -B
with consumer_scope.py, unittest discovery, then run.py prepare, stage, launch,
observe, collect and audit.py. Keep audit stdout outside the artifact directory.
Local namespace: artifacts/parakeet-observed-dense-where-graphs-amd-20260924.
VM: /dev/shm/lokad-parakeet-observed-dense-where-graphs-20260924.
