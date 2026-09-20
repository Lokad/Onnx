# Interleaved independent e5 processes: aa — September 20, 2026

The complete forty-cohort / 160-worker phase **fails
its fixed timing screens**. Complete numerical, configuration, ownership,
process and resource checks pass. All 89,088 measured,
227,583 conditioning and 5,120 solo
calls are retained. No production default changes in this report.

| Case | Policy | Boundary | A ms | B ms | C ms | ORT ms | C / mean(A,B) | C / ORT | Failed screens |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| e5-8tok | default | execute | 6.0540 | 6.0684 | 6.0728 | 6.2236 | 1.001910 | 0.975773 | B/A:visits, C/A:visits, C/B:visits, solo/resident |
| e5-8tok | default | request | 6.1229 | 6.1373 | 6.1413 | 6.2248 | 1.001822 | 0.986585 | B/A:visits, C/A:visits, C/B:visits, solo/resident |
| e5-8tok | memory | execute | 5.7436 | 5.7368 | 5.7483 | 6.1661 | 1.001401 | 0.932236 | B/A:visits, C/A:visits, C/B:visits, solo/resident |
| e5-8tok | memory | request | 5.8001 | 5.7928 | 5.8043 | 6.1673 | 1.001361 | 0.941143 | B/A:visits, C/A:visits, C/B:visits, solo/resident |
| e5-30tok | default | execute | 16.7831 | 16.6465 | 16.6560 | 15.1902 | 0.996479 | 1.096499 | B/A:aggregate, B/A:visits, B/A:positions, C/A:aggregate, C/A:visits, C/A:positions, C/B:visits |
| e5-30tok | default | request | 16.8621 | 16.7216 | 16.7318 | 15.1918 | 0.996428 | 1.101371 | B/A:aggregate, B/A:visits, B/A:positions, C/A:aggregate, C/A:visits, C/A:positions, C/B:visits |
| e5-30tok | memory | execute | 16.5139 | 16.5862 | 16.6538 | 15.3184 | 1.006267 | 1.087175 | B/A:visits, C/A:aggregate, C/A:visits, C/A:positions, C/B:visits, solo/resident |
| e5-30tok | memory | request | 16.5789 | 16.6511 | 16.7216 | 15.3201 | 1.006413 | 1.091481 | B/A:visits, C/A:aggregate, C/A:visits, C/A:positions, C/B:visits, solo/resident |
| e5-30pad128 | default | execute | 64.3904 | 64.3911 | 64.4354 | 60.2198 | 1.000693 | 1.070004 | pass |
| e5-30pad128 | default | request | 64.4652 | 64.4658 | 64.5111 | 60.2240 | 1.000707 | 1.071187 | pass |
| e5-30pad128 | memory | execute | 64.0252 | 64.1456 | 64.0381 | 59.9888 | 0.999262 | 1.067501 | pass |
| e5-30pad128 | memory | request | 64.1414 | 64.2614 | 64.1539 | 59.9929 | 0.999260 | 1.069358 | pass |
| e5-128tok | default | execute | 64.2868 | 64.3019 | 64.2384 | 59.9801 | 0.999130 | 1.070996 | pass |
| e5-128tok | default | request | 64.3621 | 64.3777 | 64.3140 | 59.9842 | 0.999132 | 1.072183 | pass |
| e5-128tok | memory | execute | 64.4987 | 64.4882 | 64.6034 | 60.4547 | 1.001705 | 1.068624 | pass |
| e5-128tok | memory | request | 64.6159 | 64.6069 | 64.7219 | 60.4588 | 1.001710 | 1.070511 | pass |
| e5-512tok | default | execute | 343.8748 | 342.1763 | 342.2487 | 280.5401 | 0.997735 | 1.219963 | B/A:visits, C/A:visits, C/B:visits |
| e5-512tok | default | request | 343.9621 | 342.2623 | 342.3374 | 280.5467 | 0.997742 | 1.220251 | B/A:visits, C/A:visits, C/B:visits |
| e5-512tok | memory | execute | 343.2431 | 347.3869 | 344.0928 | 284.0081 | 0.996461 | 1.211560 | B/A:aggregate, B/A:visits, B/A:positions, C/A:visits, C/B:aggregate, C/B:visits, C/B:positions |
| e5-512tok | memory | request | 343.3987 | 347.5414 | 344.2474 | 284.0147 | 0.996461 | 1.212076 | B/A:aggregate, B/A:visits, B/A:positions, C/A:visits, C/B:aggregate, C/B:visits, C/B:positions |

A/B use current defaults. C is identical in A/A and enables fingerprint strings
and wider LayerNorm together only in comparison. An A/A C/control ratio is not
a gain. Each engine has its own process, runtime, collector, graph and weights.
Only the active process is resumed; other resident processes are suspended.
Physical caches and memory still interact. These are resident-process timings,
not isolated-deployment estimates. The preceding isolated-deployment failure
remains unchanged.

Each worker conditions until both 128 calls and thirty cumulative execution
seconds. All first calls and conditioning are retained. Forty-eight measured
cycles comprise two locally balanced blocks of all 24 role permutations.
Batches have32/16/4/4/2calls by length. The first-created process also measures
64solo calls before others load; the last measures64after the others actually
terminate. Every role occupies both positions once per case/policy. Resident/solo
ratios across both boundaries range 0.895615 to
1.044906; every bridge remains in the observations,
including failures. Solo order and elapsed time can also affect these contrasts.

Public Execute/Run and enclosing Reset/disposal-plus-execution are measured
on every call. Pipe traffic, copying, loading and external validation are outside
the boundaries. Timing storage is allocated before measurement. No forced GC,
profiler, runtime override, fitted correction or sample exclusion is used.
Four fresh cohorts do not establish independent per-call samples or confidence
bounds. Complete visit/position screens, distributions, allocation and GC tails
are retained in the machine-readable observations.

Qualified product source4f10e8b, core `8b991fd7baaa470c45285754b20696c463dedc890a7db23dd4f0b9c7c818ccf1`. Probe/runner source
`bb1478959a3a8aedaab32561f6c18d23e1b65fbf` is a source-pinned local build against that archive.
AMD EPYC9V74 uses.NET10.0.8, AVX-512, CPU2 inherited before runtime startup;
the supervisor uses CPU0. Native ORT1.23.2 uses one thread, sequential execution,
all graph optimizations and no spinning; actual SHA256 `13ab8084954fa4a47c777880180b90810d6020f021441395712b48a75b74c68b`.
Managed workers load no native ORT. Actual startup switches are verified.

Every complete before/after output matches its retained first output, inputs
and held results stay unchanged, and all native scaled errors pass1e-4;
maximum observed error 1.63912773132e-06. Managed output bytes
match across settings and visits. Fingerprints and cache states remain valid.

All original process births are terminal. The 51,304
resource samples remain, under600seconds per cohort,12GiB combined RSS and1GiB
minimum available memory. Observed foreign CPU<=2%andgueststeal<=0.5%pass.
Inactive processes are observed stopped and their CPU counters remain stable.
Snapshots miss some short-lived work and cannot observe hypervisor neighbors.

See [README.md](README.md) for the frozen protocol, and the corresponding
`aa-observations-20260920.json` for all evidence. Artifacts are retained at
`artifacts/e5-interleaved-processes-v3-20260920`. Collection streams locally and
verifies every file. The phase receipt binds the complete evidence and reports
after writers finish; successful stages must not be rerun.

Controls failed; the candidate phase is not run.
