# Parakeet padding routes: actual workload and retained screen failures

The complete profile has **49 Pad nodes**. All 48 encoder nodes use positive
constant padding on the final axis. The remaining frontend node is reflection
padding: `nemo128.onnx`, `n6_2`, widths `[0,256,0,256]`.

The masking-candidate profile attributes **0.034416884 s** to
that frontend node, versus **2.884450168 s** to the encoder Pad
nodes. The frontend share is 0.0534%
of the profiled complete corpus. Each node has all 60 measured calls. These
profile clocks include observer overhead and are not a new application score.

All **3,840 recorded encoder Pad layouts**, covering 80 requests, 24 layers and
19 frame counts, satisfy the existing row-copy dispatcher's source guards.
These are the earlier qualified layout observations. This review does not
claim a fresh layout capture on the masking candidate or execute the dispatcher.
The reflection frontend would still use the original fallback.

The old dispatcher screen remains **rejected**: cropping, outer padding and
reflection regressed, and six repeatability controls failed. The complete
Parakeet workload does not execute its synthetic cropping or outer-padding
cases, but it does execute reflection. Its smaller measured contribution does
not establish that a changed dispatcher preserves its latency or compilation
state. That boundary needs explicit scrutiny in any future complete-model test.

This narrows the next diagnosis to encoder row copying and the actual frontend
reflection fallback. It does not admit the old prototype, change a numerical
or performance limit, or select another compiler-flag variation. Current
masking release qualification remains the prerequisite for a new experiment.

[Every observation and source identity](padding-routes-20260924.json),
[remaining complete padding groups](remaining-padding-20260924.md),
[original rejected dispatcher screen](../pad-dispatch-results/screen-20260923.md),
[retained runtime diagnosis](../pad-runtime-diagnostic-results/report-20260923.md).

No inference ran and no product changed. Closure: `eedc796f9c2982a969212e486a9e83913877320688762b90cfe8ae6b73de87ef`.
