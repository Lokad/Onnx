# Public Default and Memory options — September 19, 2026

The nine enabled CPU mechanisms reduce observed mean public Execute latency by
17–41% on the primary e5 cases compared with disabling those mechanisms on the
same current core. Explicit Memory mode substantially reduces allocation and
retained memory; its additional timing benefit varies by case. All saved
outputs match bit for bit across configurations and process visits and pass
native comparison.

These results support retaining the nine defaults and the explicit lifetime
choice. They are descriptive observations from two processes per case/setting,
not a calibrated score or proof of the 5% ORT target. This lane measures managed
configurations only; it does not reuse an older ORT mean as a fresh comparison.

| Tokens | Nine disabled, ms | Public Default, ms | Public Memory, ms | Default mean reduction |
|---|---:|---:|---:|---:|
| 8tok | 10.3503 | 6.1091 | 5.9650 | 41.0% |
| 30tok | 20.5247 | 17.0718 | 16.6496 | 16.8% |
| 30pad128 | 84.7744 | 67.0536 | 65.7458 | 20.9% |
| 128tok | 84.1340 | 65.2664 | 65.2538 | 22.4% |
| 512tok | 429.9375 | 352.5089 | 338.6620 | 18.0% |

The disabled column is an environment-controlled comparison on the same source,
not the frozen release. The [earlier matched release/ORT comparison](comparison-20260919.md)
retains its own source/settings, larger release improvements and remaining
native gap. Default remains the Speed intermediate-lifetime policy. Memory
permits earlier release of eligible dead views; inputs, initializers and held
returned outputs remain unchanged in every worker.

## Protocol and complete evidence

Frozen product source is `c6bf781816f3a45260f7a5eb33fb05d640a821e3`, built with
SDK 10.0.204. The VM is AMD EPYC 9V74, four vCPUs, 16 GiB without swap, runtime
10.0.8; every process is confined to CPU 2 before CLR startup. The first visit
runs cases 8/30/padded128/128/512, each in disabled/Default/Memory order. The
second reverses both case and configuration order. All thirty processes remain.

The disabled setting assigns exactly `0` to the nine controls documented in
[runtime options](../../docs/runtime-options.md). Default and Memory have no
experimental or runtime environment overrides. Every worker calls public
`ComputationalGraph.Execute` with the stated options, normal tiering and GC,
no forced collection and no profiling. A separate diagnostic assembly uses the
unchanged schema-4 Campaign assembly only to construct and hash canonical inputs;
it is not relabeled as that producer's historical timing protocol.

Each process records graph loading and its first execution separately, retains
all calls during 30 seconds of cumulative Execute conditioning, then records
33 Execute-only calls and a separate block of 33 Reset-plus-Execute calls.
The request block excludes tokenization and output copying. Allocation counters
cover Reset and Execute in both blocks. Block timing differences cannot be
interpreted as isolated Reset overhead because the blocks occur at different
times. No convergence filtering, sample deletion, fitted drift correction or
candidate-dependent stopping is used.

All 1,980 measured and 35,631 conditioning calls remain. Complete outputs before
and after measurement pass the pinned ORT 1.23.2 fixtures, maximum scaled error
`1.6391277313232422e-6`. Inputs are independently rehashed from complete retained
values, and the first returned tensors remain unchanged through later requests.
Every case has one output-bit identity across all configurations and visits.
Native ORT is absent from managed worker processes.

The independent analyzer verifies source/runner/core/fixture identities, fixed
order, actual affinity and runtime, sequential process boundaries, complete
output arrays, input hashes, resources and supervision accounting. Maximum
observed foreign CPU fraction is approximately `0.001434`; guest exclusivity
cannot control hypervisor neighbors. All workers terminated before collection.

## Allocation, resources and variation

| Tokens | Disabled allocation MB/request | Default allocation MB/request | Memory allocation MB/request | Default peak GB | Memory peak GB |
|---|---:|---:|---:|---:|---:|
| 8tok | 1.478 | 1.316 | 0.705 | 1.683 | 1.683 |
| 30tok | 4.136 | 3.074 | 0.841 | 1.679 | 1.681 |
| 30pad128 | 22.304 | 10.904 | 1.445 | 2.501 | 1.834 |
| 128tok | 22.304 | 10.904 | 1.445 | 2.457 | 1.796 |
| 512tok | 225.612 | 41.589 | 3.818 | 2.518 | 1.986 |

MB and GB use decimal units. Peaks include loading and the complete worker,
not just timed calls, and are finite observations. Default has ten measured
Gen2 collections across the two 512-token workers and none on primary cases;
Memory has none in any measured block. Disabled-path Gen2 tails remain in the
data, including the 512-token case. No GC time is subtracted.

Default process means range from 6.0366–6.1816 ms at 8 tokens,
16.8409–17.3027 at 30, 65.8308–68.2764 at padded128, 64.8064–65.7265 at 128,
and 348.2490–356.7687 at 512. These ranges show why the smaller Memory timing
changes should not be generalized into a precise speedup claim. At 128 tokens,
its aggregate Execute mean is almost unchanged despite much lower allocation.
Separate request-block means, all per-process samples, tails, loading and
first-call observations are retained in `summary.json` and the raw results.

## Decision and reproducibility

Retain the nine enabled mechanisms and their exact-`0` diagnostic fallbacks.
Keep public Default's existing lifetime policy and expose earlier dead-view
release through the existing explicit Memory option. Keep narrow MatMul,
interleaved GELU, wider/reciprocal softmax and other inconclusive experiments
off. The [functional/resource qualification](defaults-20260919.md) covers local
and AMD operators, native/shared models, recorded audio and package consumption;
its audio numerical failures and resource limitations remain unchanged.

The overall e5 parity and broader audio goals remain open. Another ORT verdict
requires a fresh matched native comparison with the actual chosen public
configuration. Failed earlier A/A qualification remains failed.

Ignored evidence is under `artifacts/e5-public-defaults-20260919`, including
`Program.cs`, `provenance.json`, `run.py`, raw collection and `analyze.py`.
`summary.json` retains all samples, process means, GC, allocation and resources.
SHA-256 identities:

- Core: `7653c1686419d612e2624740908a44ffdee91b4239628a979bac44f5b6b863e9`
- Frozen deployment bundle: `ef6c6e21eac37e7df1e7854157b4e7a8985a6b10fea7138ac91b8eab0ee4a979`
- Collected result archive: `bcfb6f989e6fb4d3e9b0cdfc40404de3a74f80e5843e4f4df79a97e767d1c91b`
