# Partial sums on actual Parakeet projection operands

This numerical prototype resets float accumulators every 128, 256 or 512 terms
in the captured 4,096-term projection. It differs from the preceding blocked
timing prototype, which retained the same accumulator through every block.
Product source, original captured operands and prior references are unchanged.

Four routes cover both feature sources and both convolution engines. Each
has a `[74,4096]` input and the original `[4096,1024]` weights and bias. The
frozen product kernel and one-block 4096 control must match exactly. Managed
stem controls must reproduce the original captured bits. All candidates are
checked against retained NumPy/OpenBLAS64 and Torch/MKL64 references, plus
independent scalar `fmaf` calculations of the declared partial-sum arithmetic.

Before execution, selection is fixed: every route/reference must reduce local
RMS error by at least 2x without increasing maximum scaled error or the count
above `1e-4`. Prefer 256 if it qualifies; otherwise choose the qualifying block
with the smallest worst-route RMS. Selection allows a full-model investigation,
not production promotion or a claim that the original duration failures pass.
Whole-stem error is reported separately because convolution error remains.

From the repository root, run once into the fresh artifact directory:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/reduction-accuracy/prepare.py
    C:/Python313/python.exe -X utf8 -B tests/parakeet/reduction-accuracy/run.py
    C:/Python313/python.exe -X utf8 -B tests/parakeet/reduction-accuracy/audit.py

The bounded monitor requires 4 GiB available RAM before each job, 2 GiB worker
RSS, 1 GiB available during execution, 20 GiB free disk, 1 GiB output and 600
seconds. CPU2 is inherited before .NET startup. No timing ratio is measured.
No new VM work is launched; pyannote retains first audio priority after e5.

Source motivation is the pinned ORT1.29 packed SGEMM reduction stride of 256
and FMA3 kernel partial accumulation. This does not establish which kernel the
installed native library dispatches. Source URLs and hashes are retained with
the experiment. Closed files must not be edited after execution.
