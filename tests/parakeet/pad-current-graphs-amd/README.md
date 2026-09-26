# Fresh graph regressions for the current-root padding candidate

Compare current root Core f3992f40 with padding candidate a74acb17 and fresh
Microsoft ORT 1.29.0. Require closed Parakeet correctness/application, shared/e5
correctness and complete Pyannote correctness for these exact products.
No product or consumer is built. This campaign qualifies the existing Pad change.

Run all eight existing cases: e5 at 8, 30, 30 padded to 128, 128 and 512 tokens,
DINOv3, ResNet50 and GPT-2. Preserve the original 72-job order: three numerical
processes per case, then six timed processes per case in current, candidate,
ORT, ORT, candidate, current order. For eight-token e5 use the already qualified
6,000-warmup consumer e437850d and native script. For 30-token e5 use 1,200
warmups with consumer 0b228b2d. All other cases use 600 warmups with d827e3b9.
Each timed process then measures exactly 180 calls. Keep all 73,512 clocks,
8,640 measurements and 72 setup intervals. Preserve the old failed short-e5
campaign and its diagnosed correction; neither provides new-product scores.

Require every output within the original 1e-4 scaled-error bound against fresh
native results, exact current/candidate output hashes, unchanged inputs and
independently owned held outputs. Keep every clock. Each engine's two process
means must agree within 10%; candidate/current must be <= 1.05 for every case.
The exact rational scorers and numerical/resource audits remain unchanged
apart from routing the established short-e5 consumer, script and warmup count.

The three retained compiled comparisons reverify instructions, flags, public
surface, branches, locals and exception regions. Their proof binds each reused
consumer hash; both products use the same consumer for each case. All runtime
dependencies, native scripts, models, reference tensors and tools are pinned.

Use the original monitor and limits: CPU 2 compute, CPU 0 monitoring, 11 GiB
available RAM / 3 GiB tmpfs before each job, 8 GiB owned RSS, 1 GiB remaining
RAM/tmpfs, 900 seconds per job, four hours overall, 1 GiB output per job and
2 GiB campaign files. Freeze before staging; run one VM workload at a time.
No numerical gate, timing threshold or runtime setting changes.

From repository root, prefix with `C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/pad-current-graphs-amd/consumer_scope.py
    -m unittest discover -s tests/parakeet/pad-current-graphs-amd -p test_*.py
    tests/parakeet/pad-current-graphs-amd/run.py prepare
    tests/parakeet/pad-current-graphs-amd/run.py stage
    tests/parakeet/pad-current-graphs-amd/run.py launch
    tests/parakeet/pad-current-graphs-amd/run.py observe
    tests/parakeet/pad-current-graphs-amd/run.py collect
    tests/parakeet/pad-current-graphs-amd/audit.py

Preparation, staging, launch, collection and audit run once. Observe the same
owner until terminal; do not restart because an observation times out. Keep
audit stdout outside the artifact folder. Preserve any failed gate and all
clocks; a failure permits no unchanged retry or warmup search. This result
must pass before Pyannote application staging and actual root qualification.

Local: `artifacts/parakeet-pad-current-graphs-amd-20260926`.
VM: `/dev/shm/lokad-parakeet-pad-current-graphs-20260926`.
