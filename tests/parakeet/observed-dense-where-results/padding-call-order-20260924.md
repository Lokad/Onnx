# Actual Parakeet padding call order

All 160 retained application requests run frontend reflection first, then all
48 encoder pads. Every one of the 7,840 Pad intervals is accounted for, including
warmup; decoder calls contain no Pad. Twenty complete clips precede the first
measured pass, with the same order in the selected and masking-candidate profiles.

That order predicts a substantial change in fallback warmup if the retained
row-copy dispatcher were used: the first measured frontend would follow **20**
PadCore calls instead of **980**. In the old synthetic screen, reflection's first
measurement instead follows **2,160** dispatcher-fallback calls, or **9,180**
original PadCore calls. Its reflection tensor is also a different case,
`[1,64,225]` with four values added at each end, versus the real frontend's
two-axis padding `[0,256,0,256]`.

These dispatcher counts follow source guards and previously captured layouts.
No dispatcher ran on the current product, and the retained wall trace does not
record JIT compilation events. Call counts do not identify a compilation tier
or establish the cause of a timing difference. The existing masking profiles
use the unchanged PadCore in both products.

The next padding diagnostic therefore needs the complete application order,
including all warmup requests and the first measured reflection calls. An
isolated warmed reflection kernel cannot establish that behavior. Preserve the
old screen's failed verdict; no new variant or performance admission follows
this review. Current masking release qualification remains the prerequisite.

[All 160 request clocks and predicted counts](padding-call-order-20260924.csv),
[complete chronology and source identities](padding-call-order-20260924.json),
[actual routes](padding-routes-20260924.md),
[earlier compilation observations](../pad-runtime-diagnostic-results/report-20260923.md).

No inference ran and no product changed. Closure: `c1300e9f93ae2c870a1c3845ae7065b1d280f2038ffc83c11e0991ba781c77b1`.
