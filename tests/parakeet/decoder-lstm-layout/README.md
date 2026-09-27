# One prepared LSTM layout: build and arithmetic contracts

This isolates the single intervention in
[the ORT decision](../decoder-lstm-current-review/next-decision-20260927.md).
Source control is the qualified prepared-row release a77e6f72; actual built Core
0d224bcf and Data a3745392. The release source remains unchanged.

Three isolated source files change. GraphLstmPacking.Prepare stores each existing
four-vector output group contiguously across the reduction. ColumnsPerBlock uses
the process's fixed Vector<float>.Count; arrays are neither serialized nor shared
between processes or differently configured runtimes. The new reader duplicates
LstmProjectOrdered's arithmetic and changes only addressing. The reader and group
size live in the separate internal PreparedLstmProjection class. Lstm routes its two
already-prepared projections to that reader. Scalar public execution retains its
original bypass. No new array, larger capacity, FMA, vector width, activation or
input/recurrent accumulation change is included.

The original 172 LSTM backend cases run unchanged, plus four contracts for raw
boundaries/exceptional values, every captured projection and actual product/mode
identity. All 380 captured calls supply 760 input/recurrent projections, totaling
1,945,600 compared values per hardware mode. The actual graph-prepared layout is
checked against original W/R constants and the unchanged ordered projection.
Public ownership, stale weights, scalar bypass and aggregate capacity remain
covered by the existing PreparedLstmWeightsTests. This does not replace subsequent
complete model/transcription qualification or establish a performance result.

Exactly seven serial jobs build once, compare every compiled Core/Data method,
then run contracts normally, with DOTNET_EnableAVX512=0 and with
DOTNET_EnableHWIntrinsic=0. Only GraphLstmPacking.Prepare and
CPUExecutionProvider.Lstm may change; only ColumnsPerBlock's getter and
PreparedLstmProjection.Multiply may be added. All other methods, implementation flags,
assembly attributes and public surface remain exact.

The existing dispatch-events supervisor and decoder-projection-observation
transport are reused unchanged. Work uses CPU 2; monitoring uses CPU 0. Preflight
requires 4 GiB available RAM and 1 GiB tmpfs, with 3 GiB owned RSS, 512 MiB artifact,
128 MiB per-job output and 900-second job bounds. The four-hour campaign deadline
and all owner, thread-affinity and immutable-input checks remain intact. Existing
captured fixtures are hardlinked on the VM after byte verification.

From the repository root, use Python with bytecode disabled:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/decoder-lstm-layout/run.py prepare
    C:/Python313/python.exe -X utf8 -B tests/parakeet/decoder-lstm-layout/run.py stage
    C:/Python313/python.exe -X utf8 -B tests/parakeet/decoder-lstm-layout/run.py launch
    C:/Python313/python.exe -X utf8 -B tests/parakeet/decoder-lstm-layout/run.py observe

Follow that original owner to terminal; then collect and audit once. Keep audit
stdout/stderr outside the artifact directory. Never rerun a completed namespace.
Local artifacts: artifacts/parakeet-decoder-lstm-layout-contracts-v3-amd-20260927.
VM artifacts: /dev/shm/lokad-lstmlayout3-20260927.

The first local preparation stopped before transfer or builds: the complete
parent TRX contains two pairs of duplicate truncated display names in unrelated
NpySupport cases. Its original tools, partial bundle and failure are retained
under artifacts/parakeet-decoder-lstm-layout-contracts-amd-20260927. The corrected
preparation selects the 172 LSTM cases before checking name uniqueness; all 172
remain required, with their original passing outcomes. Candidate source and
contracts are unchanged. The new contract TRX still requires uniqueness of every
result without filtering.

V2 built successfully and stopped at the compiled review before any contracts.
Adding methods inside existing types renumbered compiler-generated helpers;
assembly attributes themselves matched. Original build, products, inventory,
tools and terminal failure 5ad70f15 remain retained. V3 moves the same reader and
group-size getter to a separate internal class, preserving every original method
name. The original generic reader is untouched. No arithmetic, layout, compiler
flag or correctness threshold changes; the strict compiled scope check remains.

After exact contracts pass, freeze the component timing schedule around this
unchanged candidate and original captures. The existing plan requires at least
10% projection improvement, repeatability <=1.10 and unchanged-path regression
<=1.05. No performance claim comes from these correctness calls or worker duration.
