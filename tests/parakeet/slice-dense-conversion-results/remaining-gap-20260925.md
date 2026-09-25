# Remaining Parakeet work after the positional-copy candidate

The candidate passes the [fresh application comparison](application-20260925.md):
60.851605 seconds versus current64.941221 and ORT39.793582, a6.297410% gain
and1.529181 ORT ratio. It still needs downstream release qualification.

This review joins its retained profile to the exact earlier ORT profile. All
2,856 encoder node descriptors, edges, constants and call counts match the
original validated mapping. All217 projection groups include their original
48 fused scales. No model execution or new candidate is introduced.

| Matched work | Candidate profile (s) | Earlier ORT profile (s) | Difference (s) |
|---|---:|---:|---:|
| Constant projections, including scale | 35.449986 | 29.442083 | 6.007902 |
| Complete convolution modules | 10.094675 | 4.730615 | 5.364060 |
| Encoder padding groups | 2.923035 | 0.050175 | 2.872860 |

These rows overlap: convolution modules include padding. Do not sum them.
Profiling overhead remains included, and ORT was profiled earlier. These
differences rank diagnostic leads; they are not application gains or promised
savings. The separate fresh application comparison supplies the performance score.

The24 positional kernels now cost3.238887 seconds against ORT2.644351, leaving
0.594536 diagnostic seconds. The optimization makes repeated eligible copies
cheaper; it does not materialize one shared dense slice as ORT does.

The48 feed-forward projections with weights[1024,4096] cost12.810522 seconds
against ORT10.595551. The48 reverse projections with weights[4096,1024], including
their0.5 scales, cost12.825551 against10.636332. Together their diagnostic
difference is4.404191 seconds, larger than the remaining positional difference.
The earlier route capture found substantial per-call weight preparation on
uncached projections. That provides a focused question for the next investigation:
how much of these exact feed-forward calls is preparation versus multiplication,
and how does the installed ORT reuse its constant weights?

Do not choose a packing-budget sweep, row-tail variant or new arithmetic kernel
from these totals alone. First reconcile the actual weight retention, complete
call cost and ORT source/dispatch for these two families. Source predictions
remain distinct from sampled native execution. Finish the current candidate's
release checks before selecting another product change.

All[matched nodes, phases and source receipts](remaining-gap-20260925.json)
and the[read-only reviewer](review_remaining_gap.py) are retained. The exact
ORT binary and native routine remain documented in the
[native investigation](../ort-diagnosis-results/ort-kernels-20260924.md).
