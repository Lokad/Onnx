# Complete Parakeet correctness for the unchanged LSTM layout

Compare qualified current Core `0d224bcf` / Data `a3745392` with the already
built candidate Core `ad97b4ad` / Data `29b3f633`. Only prepared LSTM weight
layout and its same-arithmetic reader change. Reconcile all 3,981 original
Core/Data methods, flags, public bindings and metadata through the qualified
root: two Core methods change, two internal methods are added, all 697 Data
methods remain exact. No product or consumer is rebuilt.

The original v3 scalar failure remains failed. Its separate original-baseline
control reproduces the same 18 explicit-FMA rejections; all four new facts pass
in all three hardware modes, with exact captured projection hashes. The timing
screen also remains rejected: 100/168 repeatability controls fail. Subsequent
runtime observation establishes compilation within measurement, including an
LSTM OSR inside one call and competing background compilation on the same CPU.
Tracing moved the peak pass; prepared-path scatter is unresolved. Neither
diagnostic corrects the original clocks or admits an isolated speedup.

PLAN chooses an independent application decision for this one unchanged candidate
after diagnosis. This prerequisite correctness lane uses the exact retained
TranscribeReplay and AudioBenchmark binaries, original twenty clips and native
reference bytes. The five worker/validator files match pad-current-models-amd
byte for byte; audit differs only in its candidate label.

Normal and AVX512-disabled modes each check both products: 784 arrays / 3,090,494
values and twenty public transcriptions per product/mode; totals 3,136 arrays /
12,361,976 values and eighty public requests. Require original native scaled
error <=1e-4, exact integer outputs, decoder decisions, tokens, transcripts,
immutable inputs and independently held outputs. Also require bit-exact
current/candidate tensors. Agreement on fixtures is not human-label accuracy.

Preserve the original eight-job protocol, CPU2 worker / CPU0 supervisor,
11 GiB RAM / 3 GiB tmpfs preflight, 12 GiB owned RSS, >=1 GiB RAM/tmpfs remaining,
1 GiB output/job, 2 GiB stage, 1,800 seconds/job and four hours overall, including
the bounded 900-second RAM preflight wait. Verify terminal PID/birth identities
and unchanged collection receipts, then every needed runtime/asset hardlink and
external input. Completed duplicate retirement removed some historical VM source
and output copies; it preserved those receipts and all needed links. All original
proof bytes remain local. No downloads or model copies are needed.

From repository root, prefix each command with C:/Python313/python.exe -X utf8 -B:

    tests/parakeet/decoder-lstm-layout-models-amd/run.py prepare
    tests/parakeet/decoder-lstm-layout-models-amd/run.py stage
    tests/parakeet/decoder-lstm-layout-models-amd/run.py launch
    tests/parakeet/decoder-lstm-layout-models-amd/run.py observe
    tests/parakeet/decoder-lstm-layout-models-amd/run.py collect
    tests/parakeet/decoder-lstm-layout-models-amd/audit.py

Freeze sources first, execute each mutation once, observe its owner until terminal,
collect once and audit outside the artifact. Never replay completed work. Local:
artifacts/parakeet-decoder-lstm-layout-models-amd-20260927; VM:
/dev/shm/lokad-lstmlayout-models-20260927.

Successful complete correctness permits the existing independent twenty-clip
current/candidate/ORT/ORT/candidate/current comparison. Preserve one warmup, three
measurements, all 480 requests, >=1% corpus gain, <=5% per-clip regression, all
original repeatability and output gates. Broader release qualification and
explicit disclosure of the failed component screen remain required before
promotion. This lane is not a timing verdict.
