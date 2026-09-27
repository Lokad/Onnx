# Pointwise arithmetic dominates; target column remainders

The scoped observation accounts for all 3,840 pointwise calls in the original
80-request Parakeet workload. **Arithmetic consumes 95.77% of pointwise time.**
All inputs and weights already have dense storage. Input packing is only
0.027001 seconds per corpus, so a packing-layout experiment has little support.

The unchanged control takes 47.259386 seconds per corpus; the observed product
takes 47.360642, ratio 1.002143, within the predeclared 1.05 diagnostic limit.
All 160 public requests preserve exact complete results, input identities and
ownership checks. Every observed convolution belongs to its original request
interval and has one entry for each measured stage, the expected filter/sequence
geometry, and the actual `mm_unsafe_vectorized_intrinsics_2x4packed_bump` leaf.
All 48 calls per request are retained, including warmups. The table uses only
the 60 measured requests, averaged over the original three passes.

| Pointwise work | Seconds per corpus |
|---|---:|
| Matrix arithmetic | 6.215917 |
| Explicit matrix destination clear | 0.186865 |
| Output initialization, including the initial clear | 0.055821 |
| Input panel packing | 0.027001 |
| Input/weight materialization | 0.000048 |
| Remaining pointwise work, including observer bookkeeping | 0.004851 |
| **Complete pointwise calls** | **6.490502** |

These are elapsed stage measurements, with no overhead subtraction. They are
not a new release score. The qualified product remains 46.931723 seconds versus
ORT 39.513180, ratio 1.188.

## The specific arithmetic limitation

The [exact source comparison](pointwise-source-20260927.md) shows that the
ordinary Lokad leaf has eight independent accumulators for a full 32-column
block: four AVX2 vectors across two output rows. In each remaining eight-column
block, it instead has just two accumulators. The final masked block also keeps
two output rows. ORT's applicable AVX-512 routine retains its larger row block
while handling partial column panels.

The actual clocks show the expected remainder sensitivity. Averaged across
both filter counts, a 222-column call spends 10.199 ms in arithmetic, while a
225-column call spends 8.386 ms despite producing more columns. Their shapes
explain the difference: 222 has six full blocks, three eight-column remainder
blocks and one masked block; 225 has seven full blocks and one masked block.
Likewise, 88 columns take 4.932 ms and 89 take 5.679 ms when the masked loop is
introduced. These comparisons use natural workload variation, not new inputs
or kernel options.

A descriptive least-squares fit uses those exact loop counts and all 2,880
measured calls, normalizing arithmetic time by output rows. It estimates
709 ns per row for a full 32-column block, 594 ns for an eight-column remainder,
508 ns for a masked remainder and 14 ns fixed cost. Separate fits for 1,024
and 2,048 filters give similar coefficients. The pooled fit has R² 0.942 and
median absolute relative residual 1.08%. It assigns about 2.475 seconds to
column remainders, although they contain only 15.13% of output elements.
**These are model estimates, not direct timings of the individual loops.**
They support selecting a falsifiable intervention; they do not prove its gain.

## One selected intervention

Increase row sharing only in the column-remainder path of the existing two-row
packed consumer. Use eight AVX2 accumulators for eight output rows, sharing each
loaded remainder vector across them. This restores the full-block accumulator
count without changing vector width. Keep the full 32-column loop, packed bytes,
ascending reduction, input/output ownership and clearing behavior intact.
Keep FMA with the original B/A operand order for complete eight-column vectors;
keep separate multiply/add and the original operand order for masked columns.
Use the old two-row path for leftover row pairs and unsupported cases.

This is one fixed kernel intervention. Do not also change reduction blocking,
packing, flags, prefetch or the full-panel AVX-512 route. The prediction is a
larger improvement for widths with several remainder blocks, with unchanged
full-panel arithmetic. First verify every output bit, input/guard preservation,
row and column boundaries, exceptional floats and hardware fallbacks. Then
compare the affected workload and complete Parakeet requests with the original
repeatability controls. Keep the existing >=1% corpus gain and <=5% per-clip
regression admission requirements. A failed prediction needs diagnosis before
another candidate.

## Evidence and execution record

[The observation tools](../pointwise-cost-amd/README.md) are frozen at `ba33d78e`.
Only three Core methods contain observation calls; 3,283 other Core methods,
all 697 Data methods, public declarations and implementation flags match the
qualified product. The arithmetic leaf is unchanged. The observer adds 13
private methods and retains the exact application consumer.

The initial build compiled successfully but failed its final inventory command:
the reused worker supplied three arguments to a four-argument inspector.
That failed state remains code 1. Recovery `165658c7` ran only the missing
inspection against the same binaries, then produced build review `4303a321`.
No build or inference was repeated. Both capture processes subsequently exited
zero; 675 files were collected once and the full capture audit passed.

Raw artifact: `artifacts/parakeet-pointwise-cost-amd-20260927`.
Closure: `4cd3c5c3ffb4c71c40483544024426b655556b14a33e0c562304a84800e705f0`.
Observed Core: `66b6f7f39bf943f71b011167b5741a1ff52be3f83fdf4bacebc66e7f6cb23cab`.
All owners are terminal. Do not replay preparation, builds, captures or audit.

[Machine-readable totals and shape fits](pointwise-cost-observations-20260927.json)
come from the [read-only review](pointwise_cost_review.py), which verifies the
closed evidence and retains all measured calls. No candidate has been built yet.
