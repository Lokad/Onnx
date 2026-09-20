# Independent e5 deployment: aa — September 20, 2026

The fixed160-worker phase **fails its prospective timing screen**.
All complete output, configuration, cache, ownership, identity and resource
checks pass. Every 89,088 measured and 225,149
conditioning call is retained. Fresh native ORT timings use the same CPU and
inputs in separate sequential processes. No default changes in this report.

| Case | Policy | Boundary | A ms | B ms | C ms | ORT ms | C / mean(A,B) | C / ORT | Failed screens |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| e5-8tok | default | execute | 6.0883 | 6.0339 | 6.0332 | 6.1310 | 0.995398 | 0.984050 | B/A:aggregate, B/A:visits, B/A:positions, B/A:position_contrast, C/A:aggregate, C/A:visits, C/A:positions, C/A:position_contrast, C/B:positions, C/B:position_contrast |
| e5-8tok | default | request | 6.1559 | 6.1004 | 6.1004 | 6.1318 | 0.995472 | 0.994874 | B/A:aggregate, B/A:visits, B/A:positions, B/A:position_contrast, C/A:aggregate, C/A:visits, C/A:positions, C/A:position_contrast, C/B:positions, C/B:position_contrast |
| e5-8tok | memory | execute | 5.8743 | 5.8862 | 5.8566 | 6.2068 | 0.995985 | 0.943578 | B/A:visits, B/A:positions, B/A:position_contrast, C/A:positions, C/A:position_contrast, C/B:aggregate, C/B:visits, C/B:positions, C/B:position_contrast |
| e5-8tok | memory | request | 5.9289 | 5.9405 | 5.9116 | 6.2076 | 0.996114 | 0.952323 | B/A:visits, B/A:positions, B/A:position_contrast, C/A:positions, C/A:position_contrast, C/B:visits, C/B:positions, C/B:position_contrast |
| e5-30tok | default | execute | 16.8053 | 16.8124 | 16.8367 | 14.9664 | 1.001657 | 1.124966 | B/A:visits, B/A:positions, B/A:position_contrast, C/A:visits, C/A:positions, C/A:position_contrast, C/B:visits, C/B:position_contrast |
| e5-30tok | default | request | 16.8760 | 16.8830 | 16.9065 | 14.9674 | 1.001597 | 1.129555 | B/A:visits, B/A:positions, B/A:position_contrast, C/A:visits, C/A:positions, C/A:position_contrast, C/B:visits, C/B:position_contrast |
| e5-30tok | memory | execute | 16.8563 | 16.7756 | 16.5827 | 15.0865 | 0.986134 | 1.099177 | B/A:visits, B/A:positions, B/A:position_contrast, C/A:aggregate, C/A:visits, C/A:positions, C/A:position_contrast, C/B:aggregate, C/B:visits, C/B:positions, C/B:position_contrast |
| e5-30tok | memory | request | 16.9130 | 16.8327 | 16.6401 | 15.0877 | 0.986203 | 1.102891 | B/A:visits, B/A:positions, B/A:position_contrast, C/A:aggregate, C/A:visits, C/A:positions, C/A:position_contrast, C/B:aggregate, C/B:visits, C/B:positions, C/B:position_contrast |
| e5-30pad128 | default | execute | 64.9450 | 65.0917 | 64.9939 | 60.7659 | 0.999625 | 1.069580 | B/A:visits, B/A:positions, B/A:position_contrast, C/A:visits, C/A:positions, C/A:position_contrast, C/B:visits, C/B:positions, C/B:position_contrast |
| e5-30pad128 | default | request | 65.0069 | 65.1540 | 65.0568 | 60.7673 | 0.999637 | 1.070590 | B/A:visits, B/A:positions, B/A:position_contrast, C/A:visits, C/A:positions, C/A:position_contrast, C/B:visits, C/B:positions, C/B:position_contrast |
| e5-30pad128 | memory | execute | 64.6966 | 64.5605 | 64.8223 | 60.7185 | 1.002999 | 1.067587 | B/A:positions, B/A:position_contrast, C/A:positions, C/A:position_contrast, C/B:positions, C/B:position_contrast |
| e5-30pad128 | memory | request | 64.7557 | 64.6193 | 64.8809 | 60.7198 | 1.002990 | 1.068529 | B/A:positions, B/A:position_contrast, C/A:positions, C/A:position_contrast, C/B:positions, C/B:position_contrast |
| e5-128tok | default | execute | 65.1735 | 64.8532 | 64.6790 | 60.3736 | 0.994858 | 1.071314 | B/A:visits, B/A:positions, B/A:position_contrast, C/A:aggregate, C/A:visits, C/A:positions, C/A:position_contrast, C/B:visits, C/B:positions, C/B:position_contrast |
| e5-128tok | default | request | 65.2363 | 64.9164 | 64.7414 | 60.3749 | 0.994853 | 1.072323 | B/A:visits, B/A:positions, B/A:position_contrast, C/A:aggregate, C/A:visits, C/A:positions, C/A:position_contrast, C/B:visits, C/B:positions, C/B:position_contrast |
| e5-128tok | memory | execute | 64.7118 | 65.0790 | 64.7563 | 60.7765 | 0.997855 | 1.065481 | B/A:aggregate, B/A:visits, B/A:positions, B/A:position_contrast, C/A:positions, C/A:position_contrast, C/B:visits, C/B:positions, C/B:position_contrast |
| e5-128tok | memory | request | 64.7705 | 65.1377 | 64.8179 | 60.7778 | 0.997902 | 1.066473 | B/A:aggregate, B/A:visits, B/A:positions, B/A:position_contrast, C/A:positions, C/A:position_contrast, C/B:visits, C/B:positions, C/B:position_contrast |
| e5-512tok | default | execute | 354.2132 | 356.1657 | 354.9001 | 284.9228 | 0.999185 | 1.245601 | B/A:aggregate, B/A:visits, B/A:positions, B/A:position_contrast, C/A:visits, C/A:position_contrast, C/B:visits, C/B:positions, C/B:position_contrast |
| e5-512tok | default | request | 354.2803 | 356.2323 | 354.9674 | 284.9249 | 0.999187 | 1.245828 | B/A:aggregate, B/A:visits, B/A:positions, B/A:position_contrast, C/A:visits, C/A:position_contrast, C/B:visits, C/B:positions, C/B:position_contrast |
| e5-512tok | memory | execute | 342.5606 | 342.2559 | 344.8845 | 285.8061 | 1.007232 | 1.206708 | B/A:visits, B/A:positions, B/A:position_contrast, C/A:aggregate, C/A:visits, C/A:positions, C/A:position_contrast, C/B:aggregate, C/B:visits, C/B:positions, C/B:position_contrast |
| e5-512tok | memory | request | 342.6177 | 342.3129 | 344.9423 | 285.8081 | 1.007233 | 1.206902 | B/A:visits, B/A:positions, B/A:position_contrast, C/A:aggregate, C/A:visits, C/A:positions, C/A:position_contrast, C/B:aggregate, C/B:visits, C/B:positions, C/B:position_contrast |

A/B disable the cache. C also disables it in A/A, and enables it only in the
conditional comparison. Thus an A/A C/control ratio is an identical-setting
comparison, not a cache gain. Disabled processes never construct the cache.
Every process owns its graph, weights and lifetime state; no inference workers
overlap. Default and Memory are separately measured public lifetime policies.
See the [fixed protocol](README.md) and [complete observations](aa-observations-20260920.json)
for every visit, position, distribution, allocation and GC tail.

Both boundaries come from each individual call: public Execute/Run and the
enclosing Reset/disposal-plus-execution. Loading, first call and all thirty-second
conditioning observations are retained separately. Output copying and external
validation are outside timing. No forced GC, profiler, tiering override,
convergence selection or discarded observations are used. These empirical
screens are not confidence intervals or a claim of independent per-call samples.

Product is archive-qualified faf2844, core `48ca1d62ee2586d81072b8e347a671013d00abf8fe637c89eff65314e13cc710`; source-pinned probe and
runner revision `bcfee5fca4bfabb90127e12a2c6c7a2401bae9fb`. AMD EPYC9V74 uses.NET10.0.8/AVX512,
CPU2 inherited before CLR startup, supervisorCPU0. Native ORT1.23.2 uses one
intra/inter-op thread, sequential scheduling, all graph optimizations and no
spinning. Actual native SHA256 is `13ab8084954fa4a47c777880180b90810d6020f021441395712b48a75b74c68b`. Managed workers load no native ORT.

Every complete before/after array passes the unchanged1e-4 native scaled-error
gate; maximum observed error 1.63912773132e-06. Managed
output bytes match across all settings and visits. Actual first outputs and
inputs remain unchanged, as do prepared graph fingerprints and enabled cache
contents. All160workers and supervisor are terminal by original PID/birth.
All 15,122 resource samples remain. Bounds are300seconds,
6GiB group RSS,1GiB available, observed foreign CPU<=2%ofmachinecapacity and
gueststeal<=.5%. Snapshots miss some short-lived work and hypervisor neighbors.

Independent phase receipt `45455abedc9b5394b5bfbca00d028fe3b5309243a0fbc03edb67f7dd8aad5f0f` binds retained
evidence under artifacts/e5-fingerprint-deployment-20260920. Collection streams the archive locally,
without a second large VM copy. Successful workers and writers must not rerun.

Controls failed; the candidate phase is not run and the switch remains off.
