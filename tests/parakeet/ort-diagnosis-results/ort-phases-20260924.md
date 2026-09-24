# Parakeet: the installed ORT workload

This is diagnostic attribution of the original twenty-clip native application.
It preserves ORT 1.29.0, original graph outputs, all decoder decisions,
thread settings, input checks and held-output checks. No product is changed.

| Complete corpus | Boundary observer, no ORT profiler | With ORT profiler |
|---|---:|---:|
| frontend | 0.228136s | 0.231553s |
| encoder | 37.573349s | 37.961145s |
| decoder | 1.904774s | 2.108863s |
| Outside graph calls | 0.110657s | 0.113984s |
| **Complete request** | **39.816916s** | **40.415545s** |

The immediately preceding uninstrumented native comparison is 39.705483s.
Boundary observation/control is 1.002806 times that comparison;
ORT profiling is 1.015035 times the observed control.
These are separate sequential processes, so the differences include ordinary
process variation. No overhead is subtracted. Diagnostic clocks do not
replace the release benchmark or establish an optimization speedup.

Each process completes one warmup and three measured passes: 80 requests,
4,960 graph calls. Measured phases are summed over all twenty clips and
divided by three passes. Every session run is reconciled in order to an
observed graph call. Warmup remains separate. Nested operator intervals
are subtracted from parents so exclusive time is counted once.

## Observed operator cost

Times below come from the profiled process and include profiler effects.
Operator labels identify ORT graph kernels, not their internal native leaves.

| Graph | Operator | Exclusive seconds per corpus | Calls per corpus |
|---|---|---:|---:|
| encoder | MatMul | 19.639791 | 4820 |
| encoder | FusedMatMul | 10.636332 | 960 |
| encoder | Conv | 4.867332 | 1540 |
| encoder | LayerNormalization | 1.643913 | 2400 |
| decoder | LSTM | 0.995040 | 2400 |
| decoder | MatMul | 0.766738 | 2400 |
| encoder | Transpose | 0.313939 | 3920 |
| encoder | QuickGelu | 0.178199 | 1440 |
| frontend | STFT | 0.176858 | 20 |
| encoder | Add | 0.130446 | 3520 |
| decoder | FusedMatMul | 0.095309 | 1200 |
| encoder | Where | 0.094113 | 1460 |
| encoder | Pad | 0.050175 | 960 |
| encoder | Slice | 0.042268 | 1500 |
| encoder | Softmax | 0.034369 | 480 |
| encoder | Split | 0.023369 | 480 |
| decoder | Add | 0.021364 | 4800 |
| encoder | Sigmoid | 0.021302 | 480 |
| frontend | ReduceSumSquare | 0.019062 | 40 |
| encoder | Div | 0.018773 | 540 |
| encoder | Reshape | 0.014644 | 3400 |
| encoder | Mul | 0.012931 | 480 |
| frontend | MatMul | 0.012904 | 20 |
| encoder | Concat | 0.012014 | 2020 |
| encoder | Gather | 0.011860 | 2980 |
| encoder | Unsqueeze | 0.011450 | 3600 |
| decoder | Concat | 0.011283 | 2400 |
| decoder | Split | 0.009364 | 2400 |
| decoder | Squeeze | 0.008521 | 2400 |
| encoder | Shape | 0.007649 | 2000 |
| decoder | Unsqueeze | 0.006987 | 2400 |
| decoder | Transpose | 0.006301 | 2400 |
| decoder | Identity | 0.004835 | 1200 |
| decoder | Gather | 0.004743 | 1200 |
| frontend | Log | 0.004664 | 20 |
| decoder | Cast | 0.003552 | 1200 |
| decoder | Relu | 0.003438 | 1200 |
| frontend | Where | 0.003020 | 60 |
| frontend | Cast | 0.002577 | 60 |
| frontend | Slice | 0.001912 | 60 |
| encoder | Squeeze | 0.001694 | 480 |
| encoder | ReorderOutput | 0.001601 | 20 |
| frontend | Sub | 0.001415 | 60 |
| frontend | Div | 0.001019 | 80 |
| frontend | Mul | 0.000807 | 20 |
| frontend | Transpose | 0.000701 | 20 |
| frontend | Add | 0.000689 | 60 |
| frontend | Pad | 0.000640 | 20 |
| frontend | Concat | 0.000577 | 20 |
| frontend | ReduceSum | 0.000350 | 20 |
| encoder | Cast | 0.000298 | 80 |
| encoder | And | 0.000165 | 40 |
| encoder | Floor | 0.000154 | 60 |
| encoder | Not | 0.000128 | 40 |
| encoder | Sub | 0.000123 | 40 |
| frontend | Less | 0.000122 | 20 |
| encoder | Expand | 0.000116 | 20 |
| encoder | Range | 0.000104 | 20 |
| encoder | Less | 0.000100 | 20 |
| frontend | Gather | 0.000100 | 20 |
| frontend | Range | 0.000097 | 20 |
| encoder | Tile | 0.000091 | 20 |
| frontend | Unsqueeze | 0.000089 | 20 |
| encoder | Equal | 0.000079 | 20 |
| encoder | ConstantOfShape | 0.000073 | 20 |
| frontend | Shape | 0.000073 | 20 |
| frontend | Sqrt | 0.000063 | 20 |

## Identity and limits

ORT Build Info: git-branch=HEAD, git-commit-id=2e2543fbe9, fp8-kv-cache=1, build type=Release

Actual loaded native libraries are checked against the application payload.
The provider is CPUExecutionProvider, one intra/inter-op thread, sequential
execution, all graph optimizations, spinning disabled; target CPU2 and
monitor CPU0. Every request retains its original numerical/result checks.

The native operator profile does not establish exact MLAS dispatch or weight
preparation policy. [The separate graph and native instruction investigation](ort-kernels-20260924.md)
now establishes those observations and their limits. Matched Lokad phase/operator
attribution is required before reporting excess time by component.

[Every operator](ort-nodes-20260924.csv) and
[phases, profile accounting and raw-evidence identities](ort-observations-20260924.json).
All runtime shapes remain in the raw artifact analysis.json; the tracked
report avoids duplicating that16MB observation.

Raw artifact: `artifacts/parakeet-ort-diagnosis-amd-20260924`.
Closure SHA256: 615353b4d8524f8935076357c0949859553c27826d50d53ebd82fb002d146a85.
