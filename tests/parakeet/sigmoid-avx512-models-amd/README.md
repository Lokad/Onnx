# Complete Parakeet correctness for the fixed AVX-512 sigmoid candidate

Compare current qualified Core e98edee2 with candidate bdcfe20d, both with Data
7f4dd050. The sole changed original method is public Sigmoid; two private helpers
implement the fixed observed ORT arithmetic strategy. All 3,287 other original
Core methods and all 697 Data methods are unchanged, with exact original method
flags and public bindings. The 45 focused facts and all three scalar-reference
sweeps pass at closure 0953618f. Root closure 175693d3 reconciles the currently
admitted model product 8568a229 to actual root binaries. Reuse the original
TranscribeReplay and AudioBenchmark consumers and retained native references.

The five worker/validator files remain byte-exact from the original complete
model protocol; the full auditor changes only the candidate label. Each product
in normal and disabled-AVX512 modes must pass 784 arrays / 3,090,494 values and
20 complete public requests: 3,136 arrays, 12,361,976 values and 80 public requests
in total. Preserve native scaled error <=1e-4, finite values, exact integers,
decoder decisions, tokens and transcripts, bit-exact current/candidate tensors,
complete public results, immutable inputs and independently retained outputs.

Eight serial jobs use CPU2 for work and CPU0 for monitoring. Retain 11 GiB RAM /
3 GiB tmpfs preflight, 12 GiB RSS, 1 GiB RAM/tmpfs remaining, 1 GiB output/job,
2 GiB stage, 1,800 seconds/job and four hours overall. Preserve bounded waits,
failures and all original gates. No build, download or performance score.

From the repository root, use C:/Python313/python.exe -X utf8 -B with run.py
prepare, stage, launch, observe, collect; then audit.py once. Follow one owner
and never replay completed stages. Artifacts: parakeet-sigmoid-avx512-models-amd-
20260928. VM: /dev/shm/lokad-sigmoid-avx512-models-20260928.

Passing correctness permits the original six-process full application comparison,
with 63 repeatability controls and 21 gates, >=1% matched gain and no clip >5%
slower. The prospective prediction is >=0.45 seconds saved per corpus; no gain
is established yet. Root source and BENCHMARK.md await release qualification.
