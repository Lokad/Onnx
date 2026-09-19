# Fresh public-options / ORT comparison — September 19, 2026

Public Default's observed Execute mean is within 5% of fresh native ORT on 1 of four primary e5 cases; explicit Memory is within that margin on 1. The table below records the actual gap on the same qualified product binaries and matching inputs.

These are descriptive results from six processes per case/configuration. Historical fine A/A qualification failed; this balanced comparison does not repair that failure or establish calibrated parity, confidence bounds or a new score. All prescribed samples remain. Native timings below are fresh, not reused from an older campaign.

| Tokens | Default ms | Memory ms | ORT ms | Default / ORT | Memory / ORT |
|---|---:|---:|---:|---:|---:|
| 8tok | 6.1477 | 5.8677 | 6.0201 | 1.0212 | 0.9747 |
| 30tok | 16.9856 | 16.6011 | 15.3596 | 1.1059 | 1.0808 |
| 30pad128 | 66.3201 | 65.2998 | 60.7387 | 1.0919 | 1.0751 |
| 128tok | 67.8665 | 65.1282 | 60.6045 | 1.1198 | 1.0746 |
| 512tok | 347.2517 | 337.0893 | 284.5694 | 1.2203 | 1.1846 |

Ratios below one favor Lokad.Onnx. The primary cases are 8, 30, 30 real tokens padded to 128, and 128 tokens. Length 512 remains required correctness/resource coverage and a parity stretch target. No release improvement is inferred by dividing these means by historical measurements.

## Protocol and correctness

The fixed schedule has ninety sequential fresh processes: six visits to each of five cases, each visiting Default, Memory and native ORT. Every case uses all six configuration permutations; each configuration appears twice in each position. Case order rotates/reverses. Inference is bound to CPU 2 before CLR startup, with supervision on CPU 0. The target is AMD EPYC 9V74, four vCPUs, 16 GiB without swap, .NET 10.0.8; build SDK is 10.0.204. The preserved VM checkout is 172181f.

Both managed settings use actual public `ComputationalGraph.Execute` and the nine qualified production defaults, without experimental/runtime overrides. Default retains the Speed lifetime policy; Memory explicitly permits earlier release of eligible dead intermediate views. Native ORT 1.23.2 uses CPU execution, all graph optimizations, one intra/inter-op thread, sequential scheduling and no spinning. The separate schema-2 `public-options-native-descriptive-v1` wrapper uses the frozen Campaign assembly only for canonical inputs/hashing; it is not the historical scorer protocol.

Each process records loading and the first call separately, retains every call during 30 cumulative Execute/Run seconds of conditioning, then measures 33 Execute/Run calls and a separate block of 33 Reset/disposal-plus-Execute/Run calls. Both blocks exclude tokenization and output copying. The actual first returned output remains held through later requests, with input/output ownership checks. No profiling, forced GC, sample deletion, convergence filtering, fitted clock correction or candidate-dependent stopping is used.

All 5,940 measured and 128,208 conditioning calls remain. Complete before/after outputs pass pinned native reference values at the unchanged scaled-error gate of 1e-4; maximum observed error is `1.6391277313232422e-06`. Managed output bits match across both settings, every visit and both phases. Inputs and held outputs remain unchanged. Reusing native reference values does not reuse any native timing. Native ORT is absent from managed worker processes.

The independent analyzer binds every extracted file to the downloaded result archive and checks complete arrays, recomputed input hashes, core/runner/native/model/tokenizer identities, fixed schedule, actual affinity/runtime, nonoverlapping process boundaries, resource guards and supervision accounting. All workers are terminal before collection. Maximum observed foreign CPU fraction is `0.004055360034102278`; guest exclusivity does not control hypervisor neighbors.

## Process variation and the complete request boundary

The following ranges contain all six within-triplet ratios of process Execute means. They are observed ranges, not confidence intervals. A triplet runs sequentially, so nearby processes can still experience different conditions.

| Tokens | Default / ORT range | Memory / ORT range | Default request ms | Memory request ms | ORT request ms |
|---|---:|---:|---:|---:|---:|
| 8tok | 0.9959–1.0430 | 0.9623–0.9912 | 6.2214 | 5.9032 | 5.9952 |
| 30tok | 1.0651–1.1512 | 1.0418–1.1166 | 16.9698 | 16.6510 | 15.3360 |
| 30pad128 | 1.0715–1.1309 | 1.0665–1.0791 | 67.2607 | 66.1308 | 62.3151 |
| 128tok | 1.0544–1.1872 | 1.0671–1.0855 | 67.6588 | 65.8334 | 61.8741 |
| 512tok | 1.2007–1.2559 | 1.1669–1.2058 | 351.9857 | 338.1565 | 286.4638 |

Request means include Reset/disposal and execution; they are separate later blocks. Subtracting the Execute block mean would not isolate Reset overhead. All per-process means, medians, tails, loading, first calls and individual durations remain in the raw results and `summary.json`.

| Tokens | Default managed MB/request | Memory managed MB/request | Default peak GB | Memory peak GB | ORT peak GB |
|---|---:|---:|---:|---:|---:|
| 8tok | 1.316 | 0.705 | 1.683 | 1.685 | 1.315 |
| 30tok | 3.074 | 0.841 | 1.678 | 1.684 | 1.516 |
| 30pad128 | 10.904 | 1.445 | 2.494 | 1.836 | 1.571 |
| 128tok | 10.904 | 1.445 | 2.476 | 1.829 | 1.604 |
| 512tok | 41.589 | 3.818 | 2.724 | 1.991 | 1.361 |

MB/GB are decimal. Allocation counters cover managed allocation; they do not measure native heap allocation. Peaks include the complete process and model loading, not just measured calls. GC counts and tails remain in both timing blocks; no collection time is subtracted. These are finite resource observations.

## Decision and identities

Retain the already qualified defaults and the explicit public lifetime choice. This comparison introduces no product code, numerical tolerance, runtime setting or new default. Any subsequent kernel candidate must be measured against these current binaries with its own correctness and complete-model evidence. The [prior managed-only comparison](public-options-20260919.md), [functional qualification](defaults-20260919.md) and [historical release/experimental comparison](comparison-20260919.md) keep their original source/settings and limited conclusions.

Product source: `c6bf781816f3a45260f7a5eb33fb05d640a821e3`. Core SHA256: `7653c1686419d612e2624740908a44ffdee91b4239628a979bac44f5b6b863e9`. Native library SHA256: `13ab8084954fa4a47c777880180b90810d6020f021441395712b48a75b74c68b`.

Bundle SHA256: `fa2af7a48ec0bc5ab7227f981e1bfc38cdfdd1f7663f76312254bd2593df67e8`. Collected archive SHA256: `c993fc2a093cafa6bdbab0a717a34fbfa688a6634052e740af0bbe2542bda24e`. Independent receipt SHA256: `329fae1dccfddbd43d73da14c4c0ff2bbf4e6fca22a8da53ec6ad0231eaf50aa`.

The immutable wrapper source, protocol, fixed schedule, binaries/dependency identities, supervision records, full arrays, raw timings and independent analyzer are retained in `artifacts/e5-public-ort-20260919`. Its collector, analyzer and receipt writer are complete and must not be rerun over existing evidence.
