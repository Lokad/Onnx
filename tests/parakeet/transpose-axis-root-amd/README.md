# Qualify the measured transpose change as the release package

This adapter must bind actual graph and complete Pyannote application admissions
before changing root source. Parakeet application 2e4741a6 already admits a
2.420895% matched gain: 43.459338592 seconds versus current 44.537545594 and
ORT 39.202810428, ratio 1.108577118. All 63 repeatability controls and 21 gates
pass. The prospective saving of at least 0.45 seconds is met; parity <=1.05
remains open. Do not ascribe the entire gain to the 96 diagnosed encoder nodes
without fresh post-release attribution.

Exact export and runtime-shape diagnosis found 0.858205 seconds of excess in
four transpose families. Pinned ORT reduces these permutations to tiled matrix
transposes. The single change reuses Lokad's existing 8-by-8 tile for the same
collapsed matrices, preserving small/partial/empty faces, capabilities, types,
strides, alias protection and output ownership. Arithmetic leaves, weight
preparation and cache policy remain unchanged.

Measured candidate Core is c471f5d1 and Data b04aea50; actual current is Core
4e97e2ae with the same Data. Recovery source d311c9d1 binds all 446 qualified
root inputs, changes only TensorOps.Shape.cs / TransposeInto and adds the already
executed TransposeAxisMovementTests.cs fixture: 447 final inputs, two changed
paths. Source application backs up the old product file and verifies all other
bytes, including the source-policy test. Preserve the original failed fixture
build; its recovery changed no candidate product bytes.

Focused closure c0c063bb passes 487 checks: normal, AVX512-disabled,
intrinsics-disabled and vector-faces-disabled processes, covering all 76 actual
geometries, exceptional float bits, slices/reversed storage and ownership.
Compiled review f41cc8c3 establishes exactly one changed method, all other
3,287 Core methods and all 697 Data methods exact, with preserved flags and
public metadata. Complete Parakeet, shared/e5 and Pyannote numerical closures
8568a229, fa74521c and 3188186f also pass. No new preparation census is needed:
every preparation method and budget is unchanged in the compiled scope.

The six portable facts add six passes in each full backend mode. Require all
existing test names/outcomes, backend 3,603 passed / 43 skipped normally and
3,513 / 133 with AVX512 disabled. Tensors remain 394 / 0 in both modes. Local
Python fixtures test verifier rejection paths only and cannot qualify future
builds, test runs or performance comparisons.

Reuse all 18 original root jobs, worker, transport and package auditor. Build
once on the AMD VM with SDK 10.0.204, the existing offline feed and --tl:off.
All 3,288 Core and 697 Data methods, flags, public declarations and assembly
attributes must match the measured candidate. The independent NuGet consumer
must load that actual Core; the package has only Google.Protobuf 3.33.5 as a
dependency. Retain the warning census, tensor-source archive and original
PackageReference numerical, import, prepared-call and ownership checks.

Verify terminal owners and every retained prerequisite input. Use resolved,
hash-verified copies where historical evidence lives on a different filesystem.
Keep CPU2 compute / CPU0 monitor, 10GiB RAM / 3GiB tmpfs preflight, 8GiB owned
RSS, 1GiB remaining RAM/tmpfs, 900 seconds/job and four hours overall. No Windows
.NET builds or new downloads. Check allocation before collecting large outputs.

From repository root prefix C:/Python313/python.exe -X utf8 -B:

    tests/parakeet/transpose-axis-root-amd/source_scope.py
    tests/parakeet/transpose-axis-root-amd/consumer_scope.py
    -m unittest discover -s tests/parakeet/transpose-axis-root-amd -p test_*.py

After actual graph/application admission, bind digests, freeze the adapter, and
run apply_integration.py, then run.py prepare, stage, launch and observe in this
directory. Follow the same owner to terminal; collect once and audit.py once,
with audit console output outside the artifact. Preserve failures. Do not replay
completed stages. Promote source and BENCHMARK.md only after actual qualification.

Artifact: artifacts/parakeet-transpose-axis-root-amd-20260928.
Integration: artifacts/parakeet-transpose-axis-root-integration-20260928.
VM: /dev/shm/lokad-transpose-axis-root-20260928.
