# Where the current attention difference occurs

The qualified Parakeet application remains **45.332654 seconds versus ORT
39.232133, ratio 1.155498**. This read-only diagnosis changes no release result.

All 9,600 attention calls (7,200 measured) reconcile to the fresh full profiles.
The following figures are seconds per complete 20-clip corpus, averaged over
three measured passes. Observation overhead remains included.

| Retained preparation | Weights | Measured calls | Lokad | ORT | Difference |
|---|---:|---:|---:|---:|---:|
| Available and row guard accepts | 28 | 957 | 1.024309 | 0.953813 | 0.070495 |
| Available but row guard declines | Same 28 | 723 | 1.126546 | 0.863467 | 0.263079 |
| Unavailable | 92 | 5,520 | 7.222698 | 6.131573 | 1.091125 |
| Total | 120 | 7,200 | 9.373553 | 7.948853 | 1.424700 |

The 92 unprepared weights account for 76.6% of the attention difference. This
narrows the next question to their per-call packing, main multiplication and
final-row costs. The complete-call difference does not identify which of these
causes the excess. Increasing a packing budget or changing a kernel is not yet
justified; earlier rejected global trials remain closed.

[The source review](next-diagnosis-20260927.md) traces ORT's preparation of
constant B and its prepared MLAS traversal. The current native profile omits B
from the runtime inputs of all 120 nodes, consistent with that source path.
This does not provide a fresh native assembly sample for each node.

The managed route labels are historical observations revalidated against the
current source, unchanged encoder descriptors and weights, preparation policy,
owned-weight census and all actual shapes. The few intervening source edits are
reversed in memory to exact original text, and their changed branches cannot
apply to these square attention weights. They are deductions for the current
product, not new internal-route samples. Historical clocks and tensor-copy
costs are excluded; every timing above comes from the new profiles.

The reviewer at commit 116773cd passes two real-data tests covering poisoned
historical clocks and rejection of missing, duplicated or changed metadata.
Closure e83b34fb3aef0dacfc387c919638fd710fa266cd11a315fb6494195cd29ebc72
binds [all evidence and totals](attention-routes-20260927.json) and
[every raw clock](attention-clocks-20260927.csv). No inference was run.

Next, observe only the missing stage costs on an isolated copy of the qualified
product, keeping its arithmetic leaves, Data assembly and complete application
consumer unchanged. Require complete request/call accounting and an unchanged
control with at most 5% observation overhead before using the split to select
one causal intervention. These diagnostic times must not enter BENCHMARK.md.
