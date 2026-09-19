# Packed-weight alignment experiment — September 19, 2026

Explicit 64-byte alignment does not justify a production storage change. The thirty-row bank is 0.29% slower in aggregate; the 128-row bank improves 0.81%, with mixed worker results overlapping control movement. Isolated projections are also mixed. Keep the qualified current allocation and arithmetic; do not start a full-model campaign for this candidate.

ORT's pinned 1.23.2 source allocates packed GEMM weights through its CPU allocator, which honors the MLAS preferred alignment of 64 bytes. Lokad's current GraphPacking creates exact-sized float arrays. Two fresh actual e5 runs on the AMD VM found this address distribution across all 72 packed matrices / 84,934,656 bytes:

| Address remainder modulo 64 | 0 | 8 | 16 | 24 | 32 | 40 | 48 | 56 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Matrices, both workers | 9 | 10 | 11 | 11 | 8 | 7 | 7 | 9 |

The distribution and addresses remained unchanged after the second request. Every packed/source hash matches the Windows observation. Complete before/after e5 outputs pass the native gate, maximum scaled error 1.1920928955078125e-6; repeated output bits agree within each host. Cross-host output bits differ and are not required by the numerical contract. The initial geometry auditor mistakenly required them; its failing source is preserved separately, before any receipt was written.

The artifact-only timing prototype keeps every arithmetic method body from the current twelve/eight-row composer unchanged. Four owned pinned arrays provide natural allocation, alignment to 64 bytes, remainder 32, and remainder 16; the fifth mode is a duplicate using the exact natural buffer. All have identical packed values. Padded variants add 31 floats, with at least eight guard floats on either side. Pinning lasts for the bounded buffer lifetime. Natural bank allocations have 7–14 of 72 matrices aligned across workers; every actual address is saved and verified before and after use.

Ten fresh processes cover five cyclic orders and their reversals, rotated by shape. All run on CPU 2, AMD EPYC 9V74, .NET 10.0.8, normal tiering/GC and no forced collection. Complete-call timing includes destination clearing and computation. Packing is outside timing; allocation and packing durations remain separate observations. The bank has 72 matrices per mode, larger than the 32-MiB L3, and does not reproduce layer activation dependencies. These are kernel diagnostics, not model measurements.

| Rows | Duplicate / natural | Aligned / natural | Remainder 32 / natural | Remainder 16 / natural | Aligned worker range | Duplicate worker range |
|---|---:|---:|---:|---:|---|---|
| 8 | 0.9402 | 0.9220 | 0.9311 | 0.9265 | 0.5401–1.0387 | 0.5362–1.1971 |
| 30 | 0.9976 | 1.0029 | 0.9903 | 0.9921 | 0.9749–1.0661 | 0.9807–1.0147 |
| 128 | 0.9991 | 0.9919 | 0.9959 | 0.9968 | 0.9699–1.0125 | 0.9736–1.0212 |
| 512 | 1.0007 | 0.9958 | 0.9962 | 0.9979 | 0.9832–1.0057 | 0.9886–1.0155 |

Aggregate ratios divide the arithmetic means of all retained samples; worker ranges use each worker's seven-sample mean. Lower favors the named mode. No sample was removed. In particular, worker 7's eight-row natural block contains 4.9922–9.5067 ms samples where other modes average about 4 ms; worker 8's duplicate block also contains two large samples. Thus the apparent aggregate eight-row gain accompanies a large gain in an unchanged control. Its cause is not established. Zero GC collections were observed across recorded warmup/measurement intervals; that does not rule out other runtime or platform effects.

| Isolated shape: rows × reduction × columns | Aligned / natural | Duplicate / natural | Aligned worker range |
|---|---:|---:|---|
| 8 × 384 × 384 | 0.9979 | 0.9954 | 0.9649–1.0292 |
| 8 × 384 × 1536 | 0.9959 | 0.9984 | 0.9798–1.0096 |
| 8 × 1536 × 384 | 1.0042 | 1.0018 | 0.9747–1.0691 |
| 30 × 384 × 384 | 0.9844 | 0.9969 | 0.9417–1.0209 |
| 30 × 384 × 1536 | 1.0009 | 1.0010 | 0.9590–1.0608 |
| 30 × 1536 × 384 | 0.9942 | 1.0026 | 0.8919–1.1098 |
| 128 × 384 × 384 | 0.9998 | 1.0021 | 0.9819–1.0260 |
| 128 × 384 × 1536 | 1.0027 | 0.9503 | 0.6452–1.5773 |
| 128 × 1536 × 384 | 1.0151 | 1.0025 | 0.9755–1.1230 |
| 512 × 384 × 384 | 1.0006 | 0.9989 | 0.9827–1.0235 |
| 512 × 384 × 1536 | 1.0148 | 1.0041 | 0.8326–1.3350 |
| 512 × 1536 × 384 | 0.9969 | 0.9989 | 0.9733–1.0259 |

Each worker passes 256 finite/exceptional geometry cases and eight refused geometries. Checks include complete output bits, independent scalar-FMA coordinates, nonzero and repeated accumulation, sliced input/output guards, packed guards and unchanged source inputs. Every mode uses the same arithmetic; the actual source comparison confirms that only namespace/class visibility and a static import differ from the production source. The local AVX2 host checks unsupported refusal and all allocation/offset/packing contracts; full arithmetic proof runs on AMD.

The independent audit verifies the complete archive, source/binary identities, exact cases, all five actual buffer layouts, stable bank addresses, identical packed/output hashes, ordering, all samples and process/resource evidence. Eight additional corrupted-evidence checks pass. All 600 isolated records, 200 bank records and 5,600 samples remain. Maximum sampled process-group RSS is 490,983,424 bytes. The largest measured foreign CPU fraction is 0.00126286; process snapshots cannot account for every short-lived process. No fine statistical qualification or model speedup is claimed.

Supervisor 288113 and all ten children are terminal; geometry supervisor 287391 and its two children are also terminal. Final read-only inspection found no active dotnet process, preserved VM checkout 172181fc5ab4eb2bdc2eb7f37e80d25e482a0887, and 92,368,896 free disk bytes. No product source, default, tolerance or cache accounting changed.

The frozen core remains SHA256 `7653c1686419d612e2624740908a44ffdee91b4239628a979bac44f5b6b863e9`. Geometry receipt SHA256: `606e5a5aa6c8f30d3bdbf9d6341b54745e34156268aab6e399dfa423f57ee142`. Timing bundle SHA256: `49c5d269404f304dc4fff728a06049f958681c7f777a3df6ababda706280ccd4`. Collected timing archive SHA256: `38408b6c414b49f82d18998f5e99b2ee330ea14d872380512d26222914e38370`. Independent timing receipt SHA256: `65caba1c65ade4dbc87fb5bdc491bc59af6be1323f34eaeadc4f9488832303a7`.

All source, exact addresses, source/packed hashes, raw samples, setup costs and failure evidence remain in `artifacts/packed-alignment-20260919`. Completed preparation, collection and receipt writers are single-use. The result closes this alignment hypothesis at the tested scope; it does not establish that every alignment policy is useless on every CPU.
