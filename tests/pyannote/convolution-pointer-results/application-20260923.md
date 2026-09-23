# M28 complete Pyannote application comparison

**Not admitted for root integration.**
Complete 30-second dialogue latency is 12.959626548 s for the
current product, 12.784886460 s for the candidate and
9.022085651 s for Microsoft ONNX Runtime.
The candidate changes latency by -1.3483%; candidate / ORT is 1.417065516.
0 of twelve repeatability controls and 1 of four speed gates fail.

| Complete request | Current seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT | Candidate / current |
|---|---:|---:|---:|---:|---:|
| dialogue-30s | 12.959626548 | 12.784886460 | 9.022085651 | 1.417066 | 0.986517 |
| dialogue-0-10s | 0.602844274 | 0.594280525 | 0.431973663 | 1.375733 | 0.985794 |
| dialogue-10-20s | 0.603799261 | 0.599893604 | 0.433400866 | 1.384154 | 0.993532 |
| dialogue-20-30s | 0.598569606 | 0.599097782 | 0.433340987 | 1.382509 | 1.000882 |

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
application gates were fixed before M28 component timing. The separate
application parity target remains Lokad/ORT <=1.05.

Failed criteria:

- Speed, dialogue-30s: 0.986516580, limit 0.97.

Fresh native conformance passes four Pyannote requests and twenty Parakeet
clips. All 64 managed timing results equal fresh current results exactly,
including across processes. Complete native timelines, centroids,
transcripts, read-only inputs and held-output checks pass.

Both 600-second meetings (ES2004a and IS1009a) and the 30-second recovery
pass all original comparisons, with maximum centroid error 9.24801830268e-07.
Their diagnostic clocks are separate from the timing table. Previously
closed full-product, Pyannote, Parakeet and shared/e5 qualifications are
verified as prerequisites, without repeating their completed checks.

All ten jobs and 2,240 resource observations pass; peak owned RSS is
2,809,192,448 bytes. Every recorded owner is terminal. Process snapshots and
foreign-CPU accounting pass, retaining the known limitation that snapshots
can miss short-lived processes. Repeatability gates remain mandatory.

Closure: `ae998ef9106ce143e26e83a856f8575af4bff1240049ecdd08edbfa57501d54b`.
Payload: `9974c86ca446cccd30506a1032830afbe7ccdb1e9e6cf1bcc0f6d7a94cfe1b7d`.

[Every timing clock](application-clocks-20260923.csv),
[every setup interval](application-setup-20260923.csv),
[all controls, identities and resources](application-observations-20260923.json).

Artifacts: `artifacts/pyannote-convolution-pointer-app-amd-20260923`.

Selected Core: `208371f620fcfce5ff709db54525fa158b4c9442685430afbeb730619e1ded81`.
Data: `b935837024c3ffd67e322ba1fe44405b24b7da221779cf00856f090df08693f5`.

Candidate Core: `e776cec27d52baac1c0975949a128057938328c4e866562817a16865cecc3f07`.
Data: `0c55b6503c5321d01aceee132fe9ce1fed14c8be270d048de22a73b4126e5064`.

The selected root is unchanged. No unchanged timing retry or integration follows this verdict.
This campaign supplies no new matched Parakeet timing ratio.
