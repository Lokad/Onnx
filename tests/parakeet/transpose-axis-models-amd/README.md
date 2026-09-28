# Complete Parakeet correctness for tiled axis movement

Compare qualified current Core 4e97e2ae with candidate c471f5d1, both using
unchanged Data b04aea50. Only Tensor<T>.TransposeInto differs. All arithmetic
leaves, weight-preparation policy, original flags and public bindings remain
unchanged. Focused closure c0c063bb passes 487 bit/layout/ownership/fallback and
source-policy checks. Reconcile root ee4a38ff with the admitted model product
at 2d64fad9, then bind the original TranscribeReplay and AudioBenchmark consumers.
No rebuild, new model download, kernel search or performance score.

The five original worker/validator files remain byte-exact from
decoder-lstm-layout-models-amd/pad-current-models-amd; the complete auditor changes
only the candidate label. Each product in normal and disabled-AVX512 mode must
pass 784 arrays / 3,090,494 values and twenty complete public transcriptions.
Totals: 3,136 arrays, 12,361,976 values and eighty public requests.

Require native scaled error <=1e-4, finite values, exact integers, decoder
decisions, tokens and transcripts. Also require bit-exact current/candidate
tensors, complete public results, immutable inputs and independently held
outputs. Keep every original numerical gate and preserve every failure.

Eight serial jobs use CPU2 for work and CPU0 for monitoring. Retain 11 GiB RAM /
3 GiB tmpfs preflight, 12 GiB RSS, 1 GiB RAM/tmpfs remaining, 1 GiB output/job,
2 GiB stage, 1,800 seconds/job and four hours overall. Original bounded preflight
waits and failure handling remain unchanged. Reuse retained assets only.

From repository root, prefix with `C:/Python313/python.exe -X utf8 -B` and run
`tests/parakeet/transpose-axis-models-amd/run.py` with prepare, stage, launch,
then observe. Follow the same owner until every process terminates, collect
once and run audit.py once. Artifacts: parakeet-transpose-axis-models-amd-20260928.
VM: /dev/shm/lokad-transpose-axis-models-20260928. Never replay completed stages.

Passing correctness permits the original independent application comparison:
current/candidate/ORT/ORT/candidate/current, twenty clips, one warmup and three
measured passes. Retain all clocks, >=1% matched gain, <=5% per-clip regression
and all 63 controls/21 gates. The prospective saving is 0.45 seconds; it remains
unproven. Root source and BENCHMARK.md change only after release qualification.
