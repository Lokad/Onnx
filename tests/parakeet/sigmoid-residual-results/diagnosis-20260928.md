# What the exact Parakeet comparison establishes

The qualified release remains **43.459339 seconds versus ORT 39.202810 seconds**,
or **10.86% more time**. This diagnostic changes no product or benchmark score.

ORT's observed implementation fuses 72 sigmoid/multiply pairs into alpha-1
QuickGelu and executes `MlasSiluKernelAvx512F`. Its main loop processes two
512-bit vectors, using a rational approximation and a final multiplication.
The binary, caller, instructions and native sample addresses were already
verified and rebound to this exact workload. No additional ORT run was needed.

Lokad already uses the same rational coefficients and arithmetic order. Fresh
JIT listings from both application processes show a 256-bit helper processing
eight values per iteration. Its Tier1 vector body has 50 instructions over 316
bytes, including nine FMAs, one division and five `vfixupimmps` instructions.
There is no exponential call in that vector body. This proves a generated-code
difference; instruction counts and method-level samples do not measure the cost
of each instruction or establish which compiled tier a sample executed.

## The new observation does not select a product change

Both processes used unchanged qualified Core/Data and the same application
consumer. All 160 complete request results and ownership checks passed. The
collector attached after warmup; all 120 measured boundary events are present.
The two stack exports agree, sampled request duration covers 99.974% of request
wall time, and the event stream reports zero lost events out of 676,767 records.

| Observation | Seconds per twenty-clip corpus |
|---|---:|
| Control, with the two JIT logging options | 44.621508 |
| Sampled, with the same logging options | 50.656284 |
| Sampled thread time under public Sigmoid | 1.737972 |
| Of that: rational arithmetic helper | 0.738387 |
| Of that: samples stopping in public Sigmoid | 0.999540 |
| Of that: other managed descendants | 0.000044 |

The sampler adds **13.52%** to this control. The logging control is **2.65%**
above the previous uninstrumented control; that comparison also includes
between-process variation. No overhead is subtracted. These are diagnostic
observations, not a new release comparison or independently measured method CPU
time. In particular, `CPU_TIME` is the stack exporter's label.

Only **42.49%** of sampled time under Sigmoid is attributed to its rational
helper. Samples stopping in the public method do not distinguish inlined work,
native allocation/zeroing or another public-operator path. Treating all those
samples as arithmetic, or all as allocation, would be unsupported. The exported
events contain no current Sigmoid compilation event after warmup; the fresh
JIT listings and method names do not supply an instruction-to-tier join.

The apparent 8.92 seconds of request-overlapping runtime suspension also must
not be called GC cost. Matched `SuspendForGC` pairs account for **0.146054 seconds**
per corpus, and matched GC-preparation pairs for **0.001487 seconds**. Most of the
rest is `SuspendOther` during the sampled run. Fifty unmatched suspension events
remain explicitly recorded, including three GC-preparation starts. These
suspensions are not assigned to individual sigmoid nodes.

The earlier node profile measured 0.852870 seconds in the 72 sigmoid operators
and 0.115477 in their separate multiplies, versus ORT's 0.174952 for the fused
work. Those independent node clocks must not be combined with these sampled
weights to manufacture an allocation estimate or a promised fusion gain.

## Narrow next step

Inspect the retained raw stack-address and method/native mapping evidence to
resolve the public-Sigmoid samples before running another inference. First
establish whether those addresses can be recovered from this existing trace;
the current standard exports do not contain them. No vector-width, coefficient,
fusion or allocation trial is selected. If another observation is necessary,
it must answer that specific missing distinction with less observer disturbance.

After a cause is established, record its exact shapes, measured opportunity and
one saving prediction, then test one intervention with unchanged complete
application and release gates. Matrix/projection and convolution gaps remain
separate measured opportunities; this activation difference cannot explain the
whole remaining 4.26-second release gap.

Capture/export owners are terminal zero. The first local audit expected plain
JSONL, while the retained exporter emits gzip; an adapter corrected only that
reader. A report check then exposed 0.000044 seconds in other managed descendants;
the final accounting retains that third bucket. Both failures are preserved.
No inference, export or collection was repeated, and no acceptance check was
relaxed.

[Exact observations and input hashes](observations-20260928.json),
[read-only review](review.py),
[complete preceding node attribution](../transpose-axis-profile-results/diagnosis-20260928.md),
[native SiLU proof](../ort-activation-review/diagnosis-20260926.md).

Retained artifact: `artifacts/parakeet-sigmoid-residual-diagnostic-amd-20260928`.
Capture closure: `31b91fba1f513c219a7f1c4b9c44b637ff90389d676522856b64be83c5280a54`.
