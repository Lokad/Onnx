# Fixed-shape remainder experiment: not admitted

The single eight-row candidate has lower aggregate raw-kernel clocks, but the
experiment **fails repeatability and one causal-prediction gate**. It does not
establish a speedup. Preserve every clock and the original thresholds; no product
or BENCHMARK.md change is justified by this result.

The exact baseline Core `47984318` and candidate `7cac6788` run in
current/candidate/candidate/current fresh processes. Each retains five full
warmup and five measured passes over all 38 observed geometries, plus two
224-column full-panel controls. Buffers are deterministic finite synthetic data,
not captured model activations. The timer brackets only the same raw packed-kernel
delegate; packing, destination reset and verification are outside it.

All 1,600 clocks, all complete baseline output comparisons and all input/packed
immutability and zero kernel-allocation checks pass. Numerical qualification is
separate at `558f2a52`. Runtime, affinity, product identities, process accounting
and bounded resources pass. Two local auditor tests reject missing clocks,
repeatability failures, control regressions and failed predictions.

| Fixed criterion | Result |
|---|---:|
| Repeatability controls | **208 / 246 pass** |
| Shape, aggregate and prediction gates | **42 / 43 pass** |
| Sum across 38 actual shapes, baseline | 0.247870 s |
| Sum across 38 actual shapes, candidate | 0.208298 s |
| Candidate / baseline for that sum | 0.840354 |
| 1,024-row full-panel control ratio | 0.9913 |
| 2,048-row full-panel control ratio | 1.0109 |

The lower aggregate is descriptive, not an admitted 16.0% kernel gain. Thirty-six
within-process shape controls and two between-process candidate controls fail.
The largest within-process max/min ratio is 2.425. All corpus repeatability
controls pass; the uncertainty is concentrated in individual shapes.

The prediction compares 222 columns (three full remainder vectors plus a mask)
with 225 (only a mask after full panels). At 2,048 rows, the time difference
shrinks from 2.610477 ms to 0.796347 ms. At 1,024 rows, it changes from 0.924310 ms
to 0.927366 ms, failing the declared direction by about 3.1 microseconds.
Every individual shape, including both controls, passes the <=5% regression gate.
The [complete observations](pointwise-tail-timing-observations-20260927.json)
retain all 40 shape means, 246 controls, 43 gates and per-process means.

Some slow shapes occur at similar positions in both candidate processes. The
untraced records do not establish whether compilation, GC or another cause
explains them. The earlier LSTM trace cannot assign a cause to this campaign.
Do not trim slow calls, add warmups until a pass, retry unchanged, or change the
kernel based on these unstable component clocks.

Next, observe one unchanged candidate process with the same 40 shapes and
five warmup/five measured passes. Reuse the existing EventPipe collector and
exporter with per-call clock markers and GC counters, attaching before fixture
setup. Reconcile all 400 calls and 800 markers with runtime events. This is a
diagnostic of the failed screen, never a replacement score. Require evidence
before selecting a correction or an independent application decision.

Artifacts: `artifacts/parakeet-pointwise-tail-timing-amd-20260927`, closure
`b9ff0d60`. Build owner 1223343 / birth1790531159.89 and capture owner 1223629 /
birth1790531205.82 are terminal. Collections contain 49 and 69 files. All four
workers exit zero in 5.06–5.60 seconds; peak owned RSS is 446,447,616 bytes.
The original qualified Parakeet application result remains **1.188× ORT**.
