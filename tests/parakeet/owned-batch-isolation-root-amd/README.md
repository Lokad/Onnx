# Integrate and qualify the measured Parakeet candidate

Integrate only after direct-release Parakeet performance, the combined graph
qualification, shared/Pyannote correctness and Pyannote application/meeting checks
all pass. The actual baseline is Core f95a13c5 / Data a893952f; the candidate is
Core e07a4518 / Data 01e9e784. Keep product_identities admission unchanged.

source_scope.py rechecks the actual release's 427 inputs against the measured
435-input source: exactly 17 changed paths including eight additions, without
removals. It also checks the retained integration review and patch. Product
source is copied exactly; only the two prepared portable test fixtures replace
their campaign-specific versions. Root mutation is a separate, gated action.

Reuse the qualified owned-weight fixture with its nine hardware guards and the
prepared DirectDepthwiseTests fixture with seven facts covering all 59 recorded
geometries. The depthwise fixture compares exact bits against generic convolution
with two workers and retains its six other behavioral test bodies. Its source
review is complete, but it still requires compilation and execution here.
Preserve all 26 public slice-conversion cases and all existing hardware skips.

Expect complete backend outcomes of 3,546 passed / 42 skipped normally and
3,456 passed / 132 skipped with AVX512 disabled. Tensors must pass 394 cases in
each mode. Require exact test names, counts and outcomes, rejecting missing,
renamed, duplicated, failed or unexpectedly skipped cases. The Python census
tests validate these checks; they do not substitute for the actual .NET suites.

On AMD, build the actual integrated root with SDK 10.0.204/runtime 10.0.8 and
--tl:off. Require all 3,281 Core / 697 Data compiled methods, implementation flags
and public declarations to match the measured candidate. Run the original
sixteen jobs: CLI/backend/tensor builds, compiled inventory, both full suites in
both instruction modes, actual NuGet package creation, and independent
PackageReference restore/build/run. Keep Google.Protobuf 3.33.5 as the sole
dependency and preserve all import, matrix, prepared-convolution and ownership
checks. Preserve the two existing CS8604 warnings, normalizing workspace paths
only. There is no performance score in this build qualification.

Limits remain 10 GiB available RAM / 3 GiB tmpfs before execution, 8 GiB RSS,
1 GiB remaining RAM/tmpfs, 900 seconds per job and four hours total. CPU 2
computes; CPU 0 monitors. Reuse the existing offline feed. No Windows build.

Use C:/Python313/python.exe -X utf8 -B with source_scope.py and unittest discovery.
When all prerequisites pass, run integrate.py once: it checks current root bytes,
backs up existing changed files and writes the intended receipt before mutation.
Then run run.py prepare, stage, launch, observe while live, collect once terminal,
and audit.py once. Tools freeze at preparation; keep audit stdout outside the
artifact directory. Update BENCHMARK.md only after root/package qualification.

Local: artifacts/parakeet-owned-batch-isolation-root-amd-20260925.
Integration: artifacts/parakeet-owned-batch-isolation-root-integration-20260925.
Remote: /dev/shm/lokad-parakeet-owned-batch-isolation-root-20260925.
