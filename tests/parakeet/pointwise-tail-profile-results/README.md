# Diagnose the remaining Parakeet gap after the pointwise release

Source `c28b3848` is qualified at root closure `fc116763`. The independent
application result is 45.332653922 seconds versus ORT 39.232132721, a 1.155498
ratio. These fresh profiles provide attribution only and cannot update that score.

Both captures and publication are complete. Managed closure `378f7c19`, native
closure `68bb93e6` / context `6982eba8` and full partition `b7e24651` pass.
See the [current partition](diagnosis-20260927.md) and [next bounded diagnosis](next-diagnosis-20260927.md).
All owners are terminal. Do not replay the commands or overwrite publications.

The publisher requires both new captures, their actual audits and a common
qualified source, product and twenty-clip manifest. Reuse exact node membership
from mapping closure `17f0e968` / analysis `0faf7f22`, but take every clock from
the new captures. Preserve all 2,856 managed and 1,993 native encoder nodes,
other graphs, time outside nodes and time outside graphs. Reject descriptor,
shape, call-count, duplicate or coverage changes. No overhead is subtracted.

The grouping functions are identical to decoder-lstm-layout-profile-results.
Its four retained-data tests also exercise this adapter, including rejection of
changed descriptors and proof that poisoned historical clocks do not enter the
new partition. These checks perform no inference.

Only after both profile audits pass, prefix with C:/Python313/python.exe -X utf8 -B
from repository root and run once:

    tests/parakeet/pointwise-tail-profile-results/publish.py

Outputs are observations-20260927.json and diagnosis-20260927.md here, with
closure evidence in artifacts/parakeet-pointwise-tail-gap-20260927. Preserve
existing outputs and failed captures. Inspect the applicable pinned ORT source
for the largest remaining measured excess before selecting one causal change.
No further optimization is selected by the publisher.

After publication, projection_review.py splits constant projections into the
same eight reviewed families, retaining every node, fused scale edge, weight
shape and observed input shape. It reads only the fresh partition and captures,
bound to the actual root/application, and writes projection-breakdown-20260927.json.
Its node-joining and timing calculations are unchanged from the prior reviewer.
