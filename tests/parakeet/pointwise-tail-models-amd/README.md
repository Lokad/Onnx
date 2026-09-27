# Complete Parakeet correctness for eight-row column remainders

Compare qualified current Core `47984318` with unchanged candidate `7cac6788`,
both using Data `dd56902f`. One packed-matrix method changes and two private
helpers are added; all other original methods, flags and public bindings remain.
Reconcile this against the qualified root and the already tested model product.
No product or consumer is rebuilt.

The arithmetic contract passes with exact non-NaN bits and NaN classification,
while differing arithmetic NaN payloads remain explicit. This does not relax any
model-output check. Preserve the old failed numerical campaigns and the failed
timing screen. A separate unchanged-process trace finds kernel recompilation
inside a measured call and other unexplained variation; it admits no speedup.

Use the original TranscribeReplay and AudioBenchmark binaries, native reference
bytes and twenty public clips. The five worker/validator files are reused byte
for byte from decoder-lstm-layout-models-amd (and pad-current-models-amd); the
complete local auditor changes only the candidate label. Normal and
AVX512-disabled modes each test both products: 784 arrays / 3,090,494 values and
twenty public transcriptions per product/mode. Totals: 3,136 arrays, 12,361,976
values and eighty public requests.

Require native scaled error <=1e-4, finite values, exact integers, decoder
decisions, tokens and transcripts; also require bit-exact current/candidate
tensors, immutable inputs and independently held outputs. These fixtures do not
establish human-label transcription accuracy.

Retain eight serial jobs on CPU 2 with CPU 0 supervision: 11 GiB RAM / 3 GiB
tmpfs preflight, 12 GiB owned RSS, >=1 GiB RAM/tmpfs remaining, 1 GiB output/job,
2 GiB stage, 1,800 seconds/job and four hours overall. Bounded preflight waits
and failure preservation are unchanged. Reuse retained assets; no downloads.

From repository root, prefix commands with `C:/Python313/python.exe -X utf8 -B`:
`tests/parakeet/pointwise-tail-models-amd/run.py prepare`, then `stage`, `launch`
and `observe`. After that owner and children terminate, `collect` once and run
`audit.py` once. Fresh local artifacts:
`artifacts/parakeet-pointwise-tail-models-amd-20260927`; VM:
`/dev/shm/lokad-pwt-models-20260927`. Never replay completed stages.

Passing correctness permits the original independent application comparison:
current/candidate/ORT/ORT/candidate/current, twenty clips, one warmup and three
measured passes, every clock retained. Require >=1% corpus gain, <=5% per-clip
regression and all original repeatability/output gates. Root source and
BENCHMARK.md remain unchanged until application and release qualification.
