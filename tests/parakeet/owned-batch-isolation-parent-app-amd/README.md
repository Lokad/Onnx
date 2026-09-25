# Preserve Parakeet performance after dispatch relocation

Compare the successful depthwise parent Core40260aef with relocation Coree07a4518,
both using Data01e9e784. Preparation requires complete full-model correctness,
the admitted combined graph qualification and the original successful parent
application comparison. Its historical clocks are evidence, never pooled into
this new measurement. No product, runtime or consumer is rebuilt or changed.

Reuse the existing six fresh processes in parent, candidate, ORT, ORT, candidate,
parent order. Each executes the same twenty clips, one warmup and three measured
passes per clip: 480 complete requests, 120 warmups and 360 measurements. Exact
consumer hashes, model/export, audio, native settings, transcripts, tokens,
decoder decisions and owned outputs remain covered by the original validators.

This is a regression check for relocating dispatch. Apply the existing 5%
regression bound to the complete corpus as well as each clip; require every
engine's process max/min <=1.10 for corpus and <=1.20 for each clip. All 63
controls and 21 gates must pass. There is no requirement for another 3% gain
over the already optimized parent. The separate comparison against the released
product will retain its original >=3% corpus improvement requirement. Report
the actual gain or slowdown without claiming unchanged performance from bounds.

Every other worker, scorer, validator and audit rule is reused. Source-scope
checks enumerate the changed corpus-gate constant, name and policy text, its
boundary test, provenance and transport bindings. Nothing is trimmed or retried.
The independent ORT parity target remains candidate/ORT <=1.05.

CPU2 computes; CPU0 monitors. Preserve 11 GiB RAM/3 GiB tmpfs preflight, 12 GiB
RSS, 1 GiB remaining RAM/tmpfs, 1 GiB output/job, 2 GiB stage, 3,600 seconds/job
and four hours total. No overlapping VM work, downloads or local inference.

Use C:/Python313/python.exe -X utf8 -B with consumer_scope.py and
`-m unittest discover -s tests/parakeet/owned-batch-isolation-parent-app-amd -p "test_*.py" -v`.
After all prerequisites close successfully, run this directory's run.py prepare,
stage and launch once. Observe the same owner until terminal, collect and audit.py
once, keeping audit stdout outside the artifact folder. Tools freeze at preparation.

Local: artifacts/parakeet-owned-batch-isolation-parent-app-amd-20260925.
Remote: /dev/shm/lokad-parakeet-owned-batch-isolation-parent-app-20260925.
