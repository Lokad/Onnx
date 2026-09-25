# Qualify the normal M78 build and package

Integrate only after the direct release Parakeet comparison, complete graph
shortlist, shared/e5 and Pyannote correctness, and Pyannote application/meeting
gates pass. All release comparisons must use Core f95a13c5/Data a893952f as the
baseline and M78 Core 49901366/Data 01e9e784 as the candidate. The original
product_identities admission remains unchanged; an intermediate baseline fails.

source_scope.py compares the qualified release's 427 inputs against M78's frozen
433 inputs, requiring exactly 14 changed paths including six additions. M78's
isolated source preparation started from M76; do not confuse that intermediate
432-input inventory with the actual release. Integration backs up each existing
changed file before applying the exact measured source and qualified test source.
It refuses user changes and writes an intended receipt before root mutation.

The only extra source adaptation adds nine hardware guards to public owned-weight
tests. public_tests.py was already run once: its preparation receipt binds
OwnedPackedWeightTests.cs.txt and the actual 41/41/2 focused-contract TRXs.
Reversing the guards and Fact/Theory attributes must recover the entire original
text, including all assertions, helpers and InlineData. Do not rerun that generator.
The existing corrected SliceDenseConversionTests remains byte-identical.

Expect 40 new ordinary backend passes and one unavailable-hardware skip on this
FMA-capable VM. Exact complete backend outcomes are 3,539 passed/42 skipped
normally and 3,449/132 with AVX512 disabled. All 26 public slice cases join the
368 existing tensor cases: 394 passed in each mode. Keep every existing hardware
skip and require every new test's exact name, count and outcome. Do not include
campaign-specific binary identity assertions in the public suites.

Build actual root on AMD with SDK 10.0.204/runtime 10.0.8 and --tl:off. Require all
3,277 Core and 697 Data method bodies, implementation flags and public declarations
to equal measured M78. Run the original sixteen jobs: CLI/backend/tensor builds,
compiled inventory, both full test suites in both instruction modes, actual NuGet
package creation, and independent PackageReference restore/build/run. Preserve
Google.Protobuf 3.33.5 as the sole package dependency and all import, matrix,
prepared spatial/Winograd and ownership checks. Preserve the two existing CS8604
warnings and their summary repetitions; normalize only workspace paths.

consumer_scope.py proves worker, protocol, product admission, independent auditor,
warning checker and graph prerequisite are unchanged. Numerical/package checks
change only the exact method count and the declared additional test outcomes.
There is no performance measurement in this root build qualification.

Keep 10 GiB available/3 GiB tmpfs preflight, 8 GiB RSS, 1 GiB remaining RAM/tmpfs,
900 seconds/job and four hours total. CPU 2 computes; CPU 0 monitors. No local
build. With Python 3.13 -X utf8 -B, run source_scope.py and unittest discovery.
Once every prerequisite is closed and admitted, run integrate.py once, followed
by run.py prepare, stage, launch, observe while live, collect once terminal, and
audit.py once with stdout outside the campaign. Tools freeze at preparation.
Update BENCHMARK.md only after this normal root/package qualification passes.

Local artifacts: artifacts/parakeet-packed-final-row-root-amd-20260925.
Integration receipt: artifacts/parakeet-packed-final-row-root-integration-20260925.
VM: /dev/shm/lokad-parakeet-packed-final-row-root-20260925.
