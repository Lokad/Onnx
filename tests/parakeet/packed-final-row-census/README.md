# Check the real model with the proved final-row fix

Reuse the existing compiled OwnedWeightCensus consumer, with Core49901366 and
Data01e9e784 qualified at closure6295ad30. No consumer or product is rebuilt.
The source/binary bindings and all model assets are digest checked.

Require exactly87 owned feed-forward weights,37 existing prepared maps and649
initializers, unchanged payload hashes and object identities, shared execution
contexts, idempotent preparation and the exact longest public transcript in
normal and AVX512-disabled modes. These are correctness and ownership checks;
they do not measure the full corpus or score application latency.

Use Python3.13 -X utf8 -B with run.py prepare, stage, launch build, observe build,
collect build and review.py build. The build stage only binds existing files.
After that review passes, launch capture, observe capture, collect capture and
review.py capture. Collect only terminal PID/birth owners; preserve all failures
and never repeat a completed workload. Freeze tools before staging.

The unchanged guards require11GiB available RAM and2GiB tmpfs before each model
worker,12GiB owned RSS maximum,1GiB remaining RAM/tmpfs,900seconds per worker,
128MiB output,CPU2 workers andCPU0 monitoring. The qualified repository and
BENCHMARK.md remain unchanged. Full models, actual counters, matched application
comparison and retained e5 release controls follow.
