# Compare the relocation directly with the released Parakeet product

Use actual release Coref95a13c5/Dataa893952f and relocation Coree07a4518/Data01e9e784.
Retained full-model evidence covers both products in normal and AVX512-disabled
modes. A direct byte comparison checks all 1,568 tensor pairs / 6,180,988 values
and 40 complete public result pairs across those modes; no inference or timing
is performed for that comparison. Corpus and inference manifests match except
for the three product identity fields.

Preparation requires successful closure of the fresh parent comparison, complete
Parakeet correctness, exact compiled product proof and admitted combined graph
qualification. No staging or inference may overlap the parent campaign. Reuse
the existing complete application consumer, native runner, worker, scorer,
numerical validators and release-provenance auditor. Rebuild no product or consumer.

Run six fresh processes: release, candidate, ORT, ORT, candidate, release.
The twenty clips contain 213.265 seconds of audio. Each process has one warmup
and three measured passes per clip: 480 complete requests, 120 warmups and 360
measurements. Keep every transcript, token and decoder decision, owned output,
original native bound and full-call timer. No trimming, pooling across campaigns
or unchanged retry is allowed.

Keep all original 63 repeatability controls and 21 performance gates: each
engine's process max/min <=1.10 for the corpus and <=1.20 for each clip, candidate
corpus latency at least 3% lower than release, and no clip >5% slower. ORT parity
is a separate candidate/ORT <=1.05 target. The parent's retention gate does not
replace or weaken these release improvement requirements.

CPU2 computes; CPU0 monitors. Bounds remain 11 GiB RAM/3 GiB tmpfs preflight,
12 GiB RSS, 1 GiB remaining RAM/tmpfs, 1 GiB output/job, 2 GiB stage,
3,600 seconds/job and four hours total. Use the existing offline assets only.

Use C:/Python313/python.exe -X utf8 -B with consumer_scope.py and
`-m unittest discover -s tests/parakeet/owned-batch-isolation-release-app-amd -p "test_*.py" -v`.
Run release_equality.py once to bind the retained direct output comparison.
After all prerequisites pass, run this directory's run.py prepare, stage, launch,
observe while live, collect once terminal, and audit.py once. Keep audit stdout
outside the artifact folder; tools freeze at preparation. Publish the complete
verdict, including any failure. No source or BENCHMARK.md promotion follows until
shared/Pyannote and root/package qualification also pass.

Local: artifacts/parakeet-owned-batch-isolation-release-app-amd-20260925.
Remote: /dev/shm/lokad-parakeet-owned-batch-isolation-release-app-20260925.
