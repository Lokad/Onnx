# One short-e5 measurement correction after exact-product runtime diagnosis

The release f95a13c5/relocation e07a4518 comparison failed short-e5 process
repeatability. Exact-product runtime observation 5b87937a then showed that the
old 780-call prefix ends around nine seconds while optimized Lokad method loads
continue until 16.5–17.5 seconds. Preserve both the original failure and every
diagnostic clock. No product or runtime option changes in this campaign.

For eight-token e5 alone, use 6,000 fixed warmups followed by 180 measurements
for each engine. This uses the entire diagnostic extent as warmup; it does not
score a selected tail or search for a minimum passing count. All seven other
cases keep their original passing observations and measurement protocols.

The existing uninstrumented consumer changes only total calls 780 to 6,180 and
warmup 600 to 6,000. The native script changes the same two constants. The
existing worker, numerical checks and exact-clock scorer are reused. Scope
checks enumerate every adaptation. The existing compiled inspector must prove
that all other instructions, methods, flags, public interfaces, branches,
locals and exception regions are unchanged before inference starts.

Build only that consumer on the AMD VM, using SDK 10.0.204 and the retained
offline feed. Reuse the compiled inspector. Three numerical workers precede
six fresh timing workers in release, candidate, ORT, ORT, candidate, release
order. Retain all 37,089 calls, 1,080 measurements and nine setup intervals.
Original numerical, immutable-input and held-output checks remain. Require
exact candidate/release output hashes and native scaled error <= 1e-4.
Each engine's process max/min must be <= 1.10 and candidate/release <= 1.05.

CPU2 computes and CPU0 monitors. Bounds remain 11 GiB available RAM/3 GiB tmpfs
before each job, 8 GiB RSS, 1 GiB minimum free RAM/tmpfs, 1 GiB output/job,
2 GiB stage, 900 seconds/job and four hours total. No overlapping VM work.

From the repository root, prefix these commands with
`C:/Python313/python.exe -X utf8 -B`:

    tests/benchmarks/e5-steady-short-amd/scope.py
    -m unittest discover -s tests/benchmarks/e5-steady-short-amd -p "test_*.py" -v
    tests/benchmarks/e5-steady-short-amd/run.py prepare
    tests/benchmarks/e5-steady-short-amd/run.py stage
    tests/benchmarks/e5-steady-short-amd/run.py launch
    tests/benchmarks/e5-steady-short-amd/run.py observe
    tests/benchmarks/e5-steady-short-amd/run.py collect
    tests/benchmarks/e5-steady-short-amd/audit.py

Freeze tools and manifests at preparation. Observe the same owner until all
recorded processes are terminal; collect and audit once, with audit stdout
outside the artifact folder. Publish every clock and the complete verdict,
including failure. A failure stops this correction; it does not authorize a
warmup-count sweep. A passing result can support an explicit combined graph
qualification with the seven unchanged passing cases; it cannot rewrite the
original failed campaign or complete Parakeet/release qualification by itself.

Local namespace: artifacts/e5-steady-short-amd-20260925.
Remote namespace: /dev/shm/lokad-e5-steady-short-20260925.
