# Follow-up hypothesis: keep two-block dispatch outside its kernel

This is source review for possible work after M28's complete application
decision. No new candidate, measurement or integration is claimed.

The retained [M18 screen](../filter-block-screen/results-20260922.md) improved
the aggregate 2.52%, but failed its original aggregate and two per-form gates.
The two 32-output-channel forms regressed 8.36% and 6.53%. All larger eligible
forms improved, including 5.64% and 8.63% for the repeated 128-channel forms.
Those historical results retain their original verdict and are not rescored.

M18's transform inserted `m % 64 == 0` dispatch at the beginning of
`ConvBlockedSpatial.Kernel512`, calling a separate `Kernel512Four` for those
shapes. Consequently, even the retained two-block path changed. Its actual
optimized bodies were 2,140/2,146 bytes, versus 2,176/2,189 for the control.
This does not prove that dispatch or code layout caused the regression.

The [generated-code review](../filter-block-codegen/results-20260922.md)
shows **no vector spills** in either optimized reduction. The four-block
kernel uses 24 accumulators, four weights and one reused broadcast register.
Register spilling is therefore not an explanation supported by that capture.

The current caller, `ConvBlockedSpatial.Execute` in
`src/Lokad.Onnx/Zzz.ConvBlockedSpatial.cs`, selects Kernel512 or Kernel256
after geometry, alias, ISA and finite-input checks. A distinct experiment can
put the four-block selection there, leaving the qualified two-block method
byte-for-byte unchanged. Select four blocks only for 16 lanes and output
channels divisible by 64; retain the existing path for other shapes.
Geometry and the unchanged two-block packed-weight format already provide
the required full four-block weight extent for those admitted shapes.

That experiment could also apply M28's qualified row-pointer/fixed-step
address calculation to the separate four-block method, preserving the
channel/row/column FMA order, six positions and tail fusion boundary. It
would change the caller and add one private kernel, rather than inserting
another branch into the qualified two-block kernel. The interaction of
larger tiles, code growth, scalar address pressure and tiered compilation
still requires actual numerical, generated-code and complete-call checks.
The earlier gains are not additive predictions.

The [pinned ORT source review](ort-kernel-loops-20260922.md) remains relevant
to channel/weight reuse. ORT's reduction order differs; preserving Lokad's
existing ordered arithmetic is a separate requirement. Source structure
alone establishes neither installed-wheel dispatch nor a performance gain.

Before implementation, choose the selected source after M28's verdict and
freeze a new self-contained plan and acceptance rules. Keep every original
shape, fallback, repeatability and complete-application check. Do not rerun
M18 unchanged or silently promote it under later criteria.
