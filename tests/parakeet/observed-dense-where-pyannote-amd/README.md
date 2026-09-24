# Complete Pyannote qualification of the observed-mask candidate

Run only after the M70 complete Parakeet comparison is admitted. Reuse the
current release's complete Pyannote fixtures, native arrays, selected consumer
and runtime dependencies. Compare current Core/Data 37c24375 / cc37b19e
with candidate f95a13c5 / a893952f, already qualified for Parakeet.

Fresh selected and candidate processes each execute all 18 complete graph
arrays, 2,917,107 values and 16 public calls. Require original native 1e-4 bounds,
bit-identical selected tensors, complete public results including centroids,
speaker assignments, intervals, status, input immutability and held outputs.
No performance score is produced here.

The existing GraphQualification consumer contains a Data identity literal.
Build its candidate copy with only that literal updated; independently compare
all 96 compiled methods, requiring 95 unchanged and exactly that Main literal
change. Keep every prior assertion, worker, semantic test and resource limit.
The independent auditor changes only the two manifest provenance labels;
consumer_scope.py checks that scope.

Retain 11 GiB available / 3 GiB tmpfs preflight, existing per-job RSS/deadlines,
1 GiB remaining memory/tmpfs and all original output bounds. Compute on AMD
CPU 2 with CPU 0 monitoring, SDK 10.0.204/runtime 10.0.8 and .NET terminal
logging disabled. One VM workload runs at a time; use immutable hardlinks
for existing assets and unlink a target before replacing it.

Use C:/Python313/python.exe -X utf8 -B followed by consumer_scope.py,
test_semantics.py, run.py prepare, stage, launch, observe, collect,
then audit.py, with stdout outside the artifact directory. Tools freeze at
preparation. Local namespace:
artifacts/parakeet-observed-dense-where-pyannote-amd-20260924;
VM: /dev/shm/lokad-parakeet-observed-dense-where-pyannote-20260924.
