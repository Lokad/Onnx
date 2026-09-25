# Correct the short-e5 measurement before changing the product again

The exact release/relocation observation establishes that the old measured
prefix precedes substantial runtime optimization in both products. Its warmup
ends after 7.026–7.418 seconds and its measured prefix ends after 9.021–9.610
seconds. Optimized Lokad method loads continue until 16.508–17.532 seconds.
The matrix dispatchers receive their optimized versions after 12.894–15.115
seconds. Later loads include graph alias analysis and tensor enumeration.

This supports a prospective correction to a benchmark intended to measure
prepared, repeatedly executed graphs. It does not establish the cause of the
earlier uninstrumented repeatability failure. That failure remains valid for
its original protocol; no clocks are trimmed or replaced.

The entire second half of each diagnostic has no further Lokad namespace
method-load events. Across all 100 consecutive 30-call blocks in that half,
the block-mean ranges are:

| Process | Product | Minimum block (ms) | Maximum block (ms) |
| --- | --- | ---: | ---: |
| a | release f95a13c5 | 5.776331 | 5.842521 |
| b | relocation e07a4518 | 5.662834 | 5.835554 |
| c | relocation e07a4518 | 5.680719 | 5.918088 |
| d | release f95a13c5 | 5.690583 | 5.795988 |

These ranges describe runtime phases, **not an admitted performance comparison**.
All earlier blocks remain in the [complete report](report-20260925.md). A
namespace census cannot prove whole-runtime quiescence, and a loaded version
need not execute on every subsequent call. All event pairs reconcile without
unmatched compilation or suspension records.

## One prospective correction

For the eight-token e5 case alone, use **6,000 fixed warmups followed by the
original 180 measurements**, for release, relocation and ORT alike. This chooses
the complete observed diagnostic extent as warmup, rather than scoring a
favorable subsection or searching for the shortest passing count. The other
seven graph cases retain their existing protocols and qualified observations.
No product, JIT option, CPU instruction setting or acceptance threshold changes.

Adapt the existing uninstrumented benchmark by changing only its total-call
constant from 780 to 6,180 and its warmup constant from 600 to 6,000. Verify the
compiled instructions, branches, locals, exception regions and method flags
against the original consumer. Make the identical two-constant change to the
native script. Preserve original timers, numerical checks and ownership rules.

Freeze source, exact product/input identities and protocol before launch. Run
three numerical workers, then six fresh timing workers in release, candidate,
ORT, ORT, candidate, release order. Expect 37,089 total calls, including 1,080
measurements. Keep native scaled error <= 1e-4, exact candidate/release output
hashes, each engine's process repeatability <= 1.10 and candidate/release <= 1.05.
Keep the existing resource limits. Publish all clocks and any failed gate.

This is a new measurement protocol, not a retry of the old comparison. A failure
does not trigger another warmup-count search. Do not reuse diagnostic tail clocks
as measurements or alter closure def19d3f. If the new short-e5 gate passes, an
explicit combined qualification can reference it alongside the seven unchanged
passing cases in that old closure, while preserving its failed short-e5 row.
Complete Parakeet and package qualification still follow before integration.

No corrected consumer has been built, and no corrected benchmark has run.
The collected observation is closed at
`5b87937aa914ab9007928bfb3ba0f90c99eae3ff8f3d23611467cbb98ab8ed4b`;
its [full observations](observations-20260925.json) bind the exact products,
source helpers, every clock and every method-load identity.
