# Portable pyannote integration trial on AMD

This is a new prospective comparison of root production Core `d1f86a73` /
Data `e7fe1668` against current portable Core `e9c87932` / Data `85d166b5`
and fresh Microsoft ORT 1.29.0. The combined AVX-512 candidate failed its own
selection gate; its closed results and thresholds remain unchanged. Neither
that path nor the deferred-view experiment is included here.

Normal Linux builds must preserve all 3,108 Core and 697 Data compiled methods
and public declarations. Full backend (at least 3,290 passed) and tensor (342)
suites must pass, including the existing shared AVX-512 row-kernel execution
test. Prior exact-runtime AMD graph, public and Parakeet numerical qualification
is reused through pinned closed evidence: 18 pyannote arrays, 16 public calls
and 784 Parakeet arrays / 3,090,494 values per managed role. This reuse excludes
all old timing selection. Fresh native conformance adds four requests.

The exact portable build must complete ES2004a and IS1009a ten-minute meetings
and short recovery, preserving original input/ownership checks, ordinary and
exclusive native timelines and the existing centroid bound. These are new AMD
meeting executions; Windows meeting evidence is retained as a reference.

Six fresh processes run production, portable, ORT, ORT, portable, production.
Each executes one warmup and three measured passes across the full 30-second
dialogue and its three 10-second crops: **96 requests, 24 warmups, 72 measured**.
Every displayed mean retains six measured requests across two processes.
CPU2 affinity is inherited before runtime initialization, monitor CPU0;
.NET10.0.8/SDK10.0.204, unchanged original AudioBenchmark consumer and native
runner. ORT uses single intra/inter-op threads, sequential execution, full
optimization and no spinning. No runtime override or profiling accompanies timing.

Timers include features, neural inference, clustering and owned output creation;
loading, file access and external validation stay outside. Models, PCM and
all retained assertions are fixed before execution.

Performance admission requires **every** role's full-dialogue process-mean
max/min <=1.10 and every crop <=1.20. Portable full mean must be <=0.97 of
root production; each crop <=1.05. These are descriptive integration gates,
not a calibrated parity claim. Keep every sample, including warmups and slow
requests. No unchanged timing retry after a failed control or speed threshold.

Resource limits remain four hours per campaign, one hour per worker, 12GiB
owned RSS, 1GiB available memory and tmpfs free, and 2GiB artifacts. Preflight
requires 12GiB available memory and 3GiB tmpfs free. Every worker descendant and
thread is checked. Existing owners must be terminal before deployment.

From repository root, use `C:/Python313/python.exe -X utf8 -B` to run
`tests/pyannote/portable-amd-integration/prepare.py`, inspect exit0 and its
inventories, then run `finish.py` exactly once. Artifacts:
`artifacts/pyannote-portable-amd-payload-20260922` and
`artifacts/pyannote-portable-amd-execution-20260922`. Remote:
`/dev/shm/lokad-pyannote-portable-integration-20260922`.

Preparation refuses existing outputs. Frozen tools and evidence remain immutable.
Collect only after actual supervisor and worker termination; independently
recompute resource, numerical and raw-clock gates before reporting. Only an
admitted result permits applying the reviewed 23-file portable source/test patch.
Normal root build/test/package and independent consumer validation follow.
No push, publication, or ignored artifact/plan addition is part of this trial.
