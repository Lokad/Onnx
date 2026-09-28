# Lokad.Onnx 0.3.0 release

[Lokad.Onnx 0.3.0](https://www.nuget.org/packages/Lokad.Onnx/0.3.0) was published
on 2026-09-28 at 09:07:42 UTC. The local release checks pass, and **all eight
hosted CI jobs pass** on the separately committed C#/JSON fixes, `f771dcbf`.
Hosted CI tested those fixes at package version 0.2.0; the 0.3.0 package has the
local validation recorded below. The release documentation commit remains local.

## Published artifacts

The published package was regenerated from build commit
`7ede924b525926c22017d4df4fef8c4f7025c021`. Package, contents and independent
consumer smoke checks passed, and the assembly version is 0.3.0.0. NuGet's
registration confirms the version is listed. All nine original ZIP entries
match the regenerated local package byte for byte; NuGet adds `.signature.p7s`.

| Artifact | SHA-256 |
|---|---|
| Published NuGet package | `36e4b54ae6fba0ab1e65715a03df3b7458e8d07e7a45435fa2de1a2f5f26488c` |
| Local package before NuGet signing | `c11cad970f4d69ca397b84c15a4ebb8e8df602358ffef77929b247b636a84a70` |
| Local symbols package | `debe2988badde8c26ae89fa7552a582a9d74ca1ecefd7fc767f43017437c1a63` |
| Published Core assembly | `6b794a475edc6597c844bc15f57a266cc9d618dc07e983d013ac7361db66e069` |

Regeneration evidence is retained in `artifacts/nuget-regenerated-0.3.0-20260928`;
publication metadata, downloaded package and content comparison are retained in
`artifacts/nuget-publication-0.3.0-20260928`. The amended release commit records
publication in the repository changelog and documentation. The published package
retains its original build commit and packaged changelog.

## Release contents

Package, assembly and file versions are 0.3.0. CHANGELOG.md now covers the
implemented operator support, execution/memory work, audio companion projects,
qualified optimizations and remaining numerical/performance limits. It no longer
claims that every change since 0.2.0 is bitwise identical or cites superseded
preliminary performance measurements.

The NuGet package contains the managed net10.0 inference core, README, changelog,
license and icon. Its only runtime dependency is Google.Protobuf 3.33.5. Audio
orchestration APIs remain in the separate, non-packable Lokad.Onnx.Data repository
project. This release does not distribute those APIs through the core package.

## Pre-publication validation

The complete tracked checkout, including historical experiment sources, was
built on the AMD VM with .NET SDK 10.0.204. The 0.3.0 validation build passes:

| Suite | Normal passed / skipped | Intrinsics disabled passed / skipped |
|---|---:|---:|
| Backend | 3,603 / 43 | 3,323 / 323 |
| Tensors | 394 / 0 | 386 / 8 |

All 4,040 test identities are retained in each mode. Hardware and absent-model
skips remain explicit; these offline suites do not rerun the large-model lanes.
The existing admitted model and performance evidence remains applicable to the
unchanged inference methods.

Independent PackageReference consumption passes model import, byte-array and
ReadOnlyMemory import, and inference. Package content, dependency, assembly
version and binary identity checks pass. This validation package's Core is byte-identical to
the Core used by both test suites. All 3,288 Core and 697 Data method bodies,
method implementation flags and public declarations match the previously
qualified implementation. Version metadata and resulting binary identities change.
The two previously recorded CS8604 warning sites remain; this work does not
suppress them or change their execution paths.

Validation package SHA-256: `51899f71ea59380f3b8ca627db1b58ac0081f265faccecc4a2c2cfb4abe01c39`.
Validation Core SHA-256: `98a71d23cce11876aee9d5873cce6fac7d178de8980c7558e22df2ee1a490e29`.
The published package was regenerated from the same Core sources with its build
commit added to the metadata, so its assembly and package digests differ.

Evidence is retained in `artifacts/release-readiness-20260928/v030`, including
complete TRX files, resource logs, package bytes, compiled-method inventories,
input hashes, collection receipt and `analysis.json` / `closed.json`.
The audit is [eng/review-release-0.3.0.py](../eng/review-release-0.3.0.py).
The initial offline restore lacked benchmark packages; the preserved recovery
used the existing VM NuGet cache. No model download or Windows build was needed.
The initial TRX reader rejected duplicated truncated display names; the corrected
reader uses unique test IDs, preserving all cases without repeating any tests.

## Hosted CI and commit split

[GitHub run 36398477412](https://github.com/Lokad/Onnx/actions/runs/36398477412)
completed successfully on `f771dcbfac92d867470d2c1b0a306bcbd430e11c`:

| Check | Windows | Linux |
|---|---|---|
| Normal unit tests | Pass | Pass |
| Intrinsics-disabled unit tests | Pass | Pass |
| Operator differential checks | Pass | Pass |
| Package smoke | Pass | Pass |

Only this first commit, containing five C# test files and one JSON evidence file,
was pushed. The second commit contains the 0.3.0 version bump, changelog and
release documentation, including the publication record. Amendments after package
regeneration change only CHANGELOG.md, BENCHMARK.md and this report. The version
bump commit remains local, and no release tag was created by this workflow.

The fixes resolve the previous optional-parameter source-policy failure on
retained experiment inputs and 21 scalar backend failures caused by explicitly
requesting unavailable FMA instructions:

- Register 38 additional immutable experiment files under the existing exact
  path-and-content archive rule, with [closed-source evidence](../tests/Shared/closed-experiment-sources-20260928.json).
  The scanner still covers current source and test projects and rejects changed
  archive contents or any new optional declaration. Its negative checks pass.
- Preserve reversed-storage Pad and all 18 reversed-storage LSTM native cases in
  the scalar lane by selecting supported options. Explicit intrinsic and parallel
  presets are skipped where FMA is unavailable. Normal intrinsic coverage remains.

The final local 0.3.0 suites and package checks remain recorded above. Hosted CI
independently validates the C#/JSON fixes on Windows and Linux at version 0.2.0;
it does not validate the unpushed version-bump commit. Evidence for the successful
run is retained in `artifacts/release-readiness-20260928/hosted-ci-f771dcbf`.
Four GitHub API requests were used, with progress checks minutes apart. The
earlier closed local audit retains its then-current failed-CI observation; the
new run and verification receipt record the updated hosted result separately.

## Known limitations retained for this release

The [benchmark table](../BENCHMARK.md) remains the qualified implementation's
comparison: Parakeet / ORT 1.109 and Pyannote / ORT 1.108 on their stated complete
workloads. Further parity work is deferred. The rejected AVX-512 sigmoid
candidate is excluded, and no new performance experiment is running.

DINOv2 remains outside qualified timing because of numerical disagreement.
Whisper has no qualified current-release latency comparison and retains the
documented encoder/logit and broader audio accuracy limitations. See the
[model support matrix](model-support.md); these are not new regressions hidden
by the CI fixes.
