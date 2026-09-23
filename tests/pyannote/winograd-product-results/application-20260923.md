# M34 Winograd complete Pyannote application comparison

**Admitted for root integration.**
Complete 30-second dialogue latency is 13.020559423 s for the
current product, 10.453276795 s for the candidate and
9.040290939 s for Microsoft ONNX Runtime.
The candidate changes latency by -19.7171%; candidate / ORT is 1.156298715.
0 of twelve repeatability controls and 0 of four speed gates fail.

| Complete request | Current seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT | Candidate / current |
|---|---:|---:|---:|---:|---:|
| dialogue-30s | 13.020559423 | 10.453276795 | 9.040290939 | 1.156299 | 0.802829 |
| dialogue-0-10s | 0.605208017 | 0.486496160 | 0.433258157 | 1.122878 | 0.803849 |
| dialogue-10-20s | 0.609008408 | 0.482626918 | 0.435220799 | 1.108924 | 0.792480 |
| dialogue-20-30s | 0.608226840 | 0.481848468 | 0.434026299 | 1.110183 | 0.792218 |

AMD EPYC 9V74 CPU2, .NET 10.0.8 and ORT 1.29.0. Native ORT uses
one intra/inter-op thread, sequential execution, full graph optimization
and no spinning. Complete pinned native dependencies are verified.
There are no managed numerical overrides or profilers.

Six fresh processes run current, candidate, ORT, ORT, candidate, current.
One warmup and three measured passes over four fixtures retain all
96 requests: 24 warmups and 72 measurements. Setup is recorded separately.
Means reconstruct all six measured calls per fixture/role from raw ticks.
Timing includes frontend, inference and owned public results. The full
dialogue uses 21 overlapping windows; the separate ten-second crops
are different workloads, and their times are not added to reconstruct it.

Frozen controls require process-mean max/min <=1.10 for the dialogue
and <=1.20 for each crop, for every role. Candidate/current must be
<=0.97 for the dialogue and <=1.05 for each crop. Every control and
speed gate is mandatory. No samples were excluded or trimmed. These
application gates were fixed before M34 product implementation. The separate
application parity target remains Lokad/ORT <=1.05.

Fresh native conformance passes four Pyannote requests and twenty Parakeet
clips. All 64 managed timing results preserve exact speaker timelines and
decisions. Candidate centroid floats may differ from current under the
original native 1e-4 limits. Every own-product repeat remains exact,
including across fresh processes; input and held-output checks pass.
Candidate/current complete float equality: False.

Both 600-second meetings (ES2004a and IS1009a) and the 30-second recovery
pass all original comparisons, with maximum centroid error 9.72789450937e-07.
Their diagnostic clocks are separate from the timing table. Previously
closed full-product, Pyannote, Parakeet and shared/e5 qualifications are
verified as prerequisites, without repeating their completed checks.
The [retained initial shared exactness failure](shared-failure-20260923.md)
and [explicit corrected arithmetic scope](shared-20260923.md) remain
part of this qualification; no native numerical limit was relaxed.

All ten jobs and 1,931 resource observations pass; peak owned RSS is
2,817,003,520 bytes. Every recorded owner is terminal. Process snapshots and
foreign-CPU accounting pass, retaining the known limitation that snapshots
can miss short-lived processes. Repeatability gates remain mandatory.

Closure: `dc2c7b9f5086ab9b4ee615b9dad7643eaf4c1ed65c1cc71b3e76ee794237d88e`.
Payload: `60e309165247076e489357616f7a4821ac28478fc7577b1df8ce98fe36cc2951`.

[Every timing clock](application-clocks-20260923.csv),
[every setup interval](application-setup-20260923.csv),
[all controls, identities and resources](application-observations-20260923.json).

Artifacts: `artifacts/pyannote-winograd-product-app-amd-20260923`.

Selected Core: `208371f620fcfce5ff709db54525fa158b4c9442685430afbeb730619e1ded81`.
Data: `b935837024c3ffd67e322ba1fe44405b24b7da221779cf00856f090df08693f5`.

Candidate Core: `521bae1702849ca23dda586515e7cbabaac2d1eabdff04dc90a7ba76059e93fb`.
Data: `f3b9aa81ee9766797e95216714dec559c5d8b020f8df510cf5f7ee0dda82693a`.

Actual root integration and normal-root qualification remain pending.
This campaign supplies no new matched Parakeet timing ratio.
