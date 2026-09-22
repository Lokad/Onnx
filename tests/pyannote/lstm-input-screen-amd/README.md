# Fixed complete-LSTM AMD screen

Run from repository root with `C:/Python313/python.exe -X utf8 -B`:
`selftest.py`, then `run.py prepare`, `run.py stage`, `run.py launch`.
Use `run.py observe` while workers run. After all identities are terminal,
use `run.py collect` and `audit.py`. Existing destinations are refused.

The VM builds only the ordinary graph driver; selected Core `3c2f16b0` and
candidate Core `208371f6` are unchanged from exact AMD qualification. Two fresh
qualification workers first check all twelve original captured graphs twice,
all three outputs, selected bits, ORT error <=1e-4, readonly inputs, retained
outputs and exact scratch. Timing then uses four fresh processes in order
selected, candidate, candidate, selected. CPU2 affinity precedes the CLR;
monitoring runs on CPU0. No runtime flags, profiling or disassembly are enabled.

Each complete call times Reset and Execute. Per-call weight panels, bounded
input-row buffer, both projections, recurrent gates and output allocations are
included. Graph preparation/context construction have separate clocks. All
output hashes, native errors and ownership checks run after the raw clock has
been flushed. Qualification retains every output; timing retains the first
output tuple of every case throughout subsequent calls.

The geometry schedule is fixed before execution: ceil(2^31 / complete input and
recurrent MAC count), giving 19 calls for input width 60 and 10 for width 256.
All twelve captured graphs have sequence length 589, with a one-row block tail.
Other tails have separate numerical coverage. One warmup and three measured
passes give 588 calls per process: 2,352 timing clocks overall, including 588
warmups and 1,764 measurements, plus 48 preparation clocks. Qualification
clocks are diagnostic and never included in the score.

Every case has equal weight after averaging its measured repetitions. Aggregate
is the sum of twelve case means; node means combine the three crops equally.
Ten repeatability controls require max/min <=1.10 aggregate and <=1.20 per node.
Five speed gates require candidate/selected <=0.90 aggregate and <=1.05 per
node. A failed or incomplete screen cannot admit the product. No sample is
discarded, and an unchanged failed trial is not repeated.

The monitor enforces 12 GiB available memory / 3 GiB tmpfs before workers,
8 GiB owned RSS, 1 GiB minimum available memory/tmpfs, 900 seconds per worker,
1 GiB output and 2 GiB artifact limits. Source, schedule, inputs, runtimes and
external dependencies are pinned before workers. The independent local audit
reconstructs references and replays all accounting and scoring checks.
