# Remaining Parakeet padding cost after the masking change

The retained candidate profile still spends **2.884450 seconds**
inside all 48 encoder Pad operations, or **4.473%** of its
profiled complete corpus time. This is attribution with observer overhead,
not a new application score or an achievable speedup prediction.

| Complete padding family | Current profile (s) | Masking candidate profile (s) | Earlier ORT profile (s) |
|---|---:|---:|---:|
| attention-pad | 2.204711 | 2.186341 | 0.034881 |
| convolution-pad | 0.732805 | 0.735900 | 0.015294 |

Complete groups contain 392 unique nodes; shared input construction is counted once.
The boundaries include pad-parameter construction and exclude the preceding
data-producing matrix or convolution operation. Every layer, shape, call
and original node matches the previously qualified graph correspondence.
The ORT column is retained historical profiling of the same logical work;
it is not a fresh comparison against this candidate. No clocks are pooled.

The exact installed ORT revision dispatches four-byte elements through
`PadImpl<uint32_t>`, copies contiguous innermost rows and fills their borders.
Lokad still fills the whole output, then calculates every source element’s
destination through per-axis division and remainder. Its Pad source,
compiled bodies and existing method flags are unchanged by the masking edit.

For the observed attention `[1,8,T,2T-1]` and convolution `[1,1024,T]`
shapes, the retained source counts are 815,859,456 coordinate-loop steps
over 220,412,352 input values. ORT’s source instead traverses 1,005,504
contiguous copy blocks. These are source counts, not measured instructions
or memory traffic; allocator clearing and compiler transformations are excluded.

The earlier row-copy prototype improved its eligible synthetic shapes but
failed fallback and repeatability gates. Those failures remain rejected.
This review supports padding as the next bounded diagnosis after the current
release qualification. It does not admit that prototype, select a new variant
or authorize bypassing any correctness or application performance gate.

[All 48 operations](remaining-padding-20260924.csv),
[complete membership, source identities and counts](remaining-padding-20260924.json),
[actual layouts](../managed-phase-results/masking-layouts-20260924.md),
[earlier rejected screen](../pad-first-use-results/screen-20260923.md).

No model ran and no product file changed. Closure: `c9161db64be4f67061339195862b4680cb9e178c0125adf7ad4f73548e0ce46c`.
