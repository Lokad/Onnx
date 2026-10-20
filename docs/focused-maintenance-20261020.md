# Focused maintenance qualification, October 20, 2026

Commits `652439ad` and `642a0d1b` start from `e99fb808`. Four obsolete scripts
join the existing [Git-history archive](../archives/README.md): the old native
profile compressor, retained-artifact verifier, E5 A/B script and VM bootstrap.
All four restore byte-for-byte from the existing pinned commit; the index now
contains 5,999 unique files. The current Gemma VM bootstrap remains available.

Ordinary test builds use `eng/Tests.slnx`: Core, Data, CLI and both test projects.
Its restore graph excludes BenchmarkDotNet and native ORT. The Linux ordinary
CI job selects `eng/test.ps1 -BuildTools` to compile the full developer solution;
the other three ordinary/scalar jobs build the focused solution. CLI integration
tests remain enabled in all four jobs. Coverage remains optional.

## Allocation measurements

Immutable `ExecutionOptions` class presets are shared. `TensorExecutionOptions`
is a value type; its allocation-free preset factories are unchanged. Graph
execution combines two option copies into one and constructs node-detail
callbacks only when detailed profiling is enabled. Accounting, public metrics,
GC snapshots and profiler nesting remain supported.

Separate processes use the pinned control/candidate cores, .NET SDK 10.0.204,
disabled tiering/OSR and identical warmed loops. Five repetitions give these
median calling-thread allocation counts:

| Operation | Control bytes | Candidate bytes |
|---|---:|---:|
| Read default preset | 72 | 0 |
| Create execution context | 1,936 | 1,792 |
| One-node run, profiling disabled | 4,608 | 4,304 |
| 96-node run, profiling disabled | 53,260 | 37,754 |
| 96-node run, wall-only profiling | 61,804 | 46,298 |
| 96-node run, detailed profiling | 97,222 | 97,078 |

The small Relu graphs retain identical outputs and valid detailed/wall profiles.
The disabled and wall-only paths save approximately 160 bytes per node plus
144 bytes per run. Counts vary by up to a few bytes per run across repetitions.
These allocation results alone do not establish a throughput improvement.

## Qualification

All 2,830 public/protected declarations and 3,274 common method implementation
flags match. Normal tests pass 403 tensor and 3,599 Backend cases. The sole local
failure is the unchanged ResNet hash mismatch: expected 15780008002253672869,
actual 8566038335043033094. Scalar tests pass 395 tensor and 3,402 Backend cases
with no failures. The added test verifies that customizing presets cannot leak
reporters or pool policy into other users. All 198 Python maintenance fixtures
pass, and the full developer solution compiles. Native differential conformance
passes all 309 cases; e5 conformance passes nine. Package contents and an
independent package consumer pass; Data remains nonpackable.

The [exact source-commit CI](https://github.com/Lokad/Onnx/actions/runs/39448467254)
passes all nine jobs without retries. All 225 complete Gemma arrays match the
pinned control byte-for-byte across scalar/SIMD/default. All six control/candidate
runs pass frozen native references for 15 cases, including mixed batches and the
explicit 512-token boundary.

## Practical CPU comparison

Two balanced process pairs run serially on the existing AMD benchmark VM, with
one compute worker at a time. Gemma uses ORT 1.29.0; e5 uses its existing runner's
ORT 1.23.2. Medians are:

| Case | Control | Candidate | Candidate's ORT | Paired candidate/control |
|---|---:|---:|---:|---:|
| Gemma FP32, 128 tokens | 165.89 ms | 165.58 ms | 167.77 ms | 0.9982 |
| Gemma FP32, 512 tokens | 786.86 ms | 787.73 ms | 735.35 ms | 1.0011 |
| e5, 30 tokens | 10.13 ms | 10.28 ms | 7.68 ms | 1.0147 |

Gemma's medians differ by less than 0.2%. E5's individual pair ratios range from
1.0002 to 1.0292, with a 2.5% candidate process spread. This short engineering
check shows minor timing variation; no throughput gain or change to published
benchmark headline figures is claimed.

Independent reload validates all 24 workers and 792 timed calls, including file
bindings, pinned source/core identities, runner/native hashes, numerical checks,
raw samples, affinity and process accounting. Maximum foreign CPU fraction is
0.00177991 (0.178%). The controller and all compute workers are terminal; the
VM is idle. The 260-file evidence archive has SHA-256
`a760bb2ee147f4ee4fde374d52c3583531b5dcfa6e3b720896c9349a5e06f28d`.
Local evidence is under ignored `artifacts/focused-cleanup-20261020`.
