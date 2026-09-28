# One ORT-shaped sigmoid build

Use the completed address diagnosis a0ae70ee and isolated source 51abd218.
No product release or performance result is produced by this build.
Run run.py prepare, stage, launch build, observe build, collect build, then
review.py build. Only after that review, launch/observe/collect capture and run
review.py capture. Never repeat a completed action. Preserve a failure before
selecting a corrected successor. The existing bounded supervisor verifies source,
runtime, resource limits, CPU affinity and terminal ownership.

The original portable helper is unchanged. The only modified original method is
Sigmoid; two private helpers are added, with no API changes. Reuse qualified Data
bytes. Run fifteen focused facts in each of three processes: normal AVX-512,
AVX-512 disabled, all hardware intrinsics disabled. The existing million-value
scalar error sweep, special values, ownership, layout and scalar contracts remain.
Three new facts compare full bit patterns against the portable path, including
all 32 residues and unaligned offsets, all lanes and NaN payloads. Capture JIT
code and require the optimized two-vector rational body with 18 FMAs, two vector
divisions and no portable min/max fixups. No width/unroll/performance sweep.

The next admission is the original complete Parakeet application comparison,
with all 63 controls and 21 gates. Predict at least 0.45 seconds saved, require
at least 1% matched corpus improvement and no clip regression above 5%.
