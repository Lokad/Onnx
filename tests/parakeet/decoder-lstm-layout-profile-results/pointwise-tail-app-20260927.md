# Pointwise remainder sharing reduces complete Parakeet latency by 2.13%

The unchanged candidate passes the independent complete-transcription comparison:
**63/63 repeatability controls and 21/21 performance gates**. Across twenty clips
containing 213.265 seconds of audio, it saves 0.987812501 seconds, or 2.132562%,
against the current product in the same campaign.

| Product | Complete corpus (s) | Relative to Microsoft ORT |
| --- | ---: | ---: |
| Current `47984318` | 46.320466422 | 1.180677 |
| Candidate `7cac6788` | 45.332653922 | 1.155498 |
| Microsoft ORT | 39.232132721 | 1.000000 |

Both managed products use Data `dd56902f`. The candidate changes only the packed
kernel's column remainders to share loaded vectors across eight output rows.
Packing, reduction order and complete 32-column panels remain unchanged.

All six fresh processes completed in the original order:
current, candidate, ORT, ORT, candidate, current. Every process retained one warmup
and three measured passes. All 480 requests and 360 measured clocks are preserved;
the score gives each process equal weight. All 320 managed public results match
their qualified complete results exactly. Both native processes pass the original
native request checks. The [separate model qualification](pointwise-tail-models-20260927.md)
also passes 3,136 arrays / 12,361,976 values in normal and AVX512-disabled modes.

The two process means differ by at most 0.421% on the complete corpus across all
three engines. The largest per-clip candidate regression is 1.100356%, below the
unchanged 5% limit. All 2,131 resource samples pass; peak owned RSS is
11,325,718,528 bytes. CPU accounting, input identity, monitor continuity and
original resource limits pass. Supervisor 1226584 / birth1790534072.37 and all
children are terminal; 621 files were collected once and independently audited.

This qualifies the application gain at the measured boundary. The earlier
[component screen](pointwise-tail-timing-20260927.md) remains rejected, including
its failed prediction. The [runtime trace](pointwise-tail-runtime-20260927.md)
found recompilation inside measurement but left other variation unexplained.
Neither result has been rescored or replaced by this application verdict.

The candidate remains short of the 1.05 parity target. Shared/e5, Pyannote, graph,
portable boundary tests and actual-root/package qualification remain required
before promotion. Source review additionally identified raw widths 1–31 that
reach the new helpers but are absent from the isolated boundary fixtures; that
bounded correctness coverage must pass. Root source and BENCHMARK.md still
describe the preceding qualified release. Differences from its older campaign
must not be counted as gains from this change.

[All clocks, controls, gates, identities and resources](pointwise-tail-app-observations-20260927.json),
[application protocol](../pointwise-tail-app-amd/README.md).

Closure: `505da6ab8c9ce1d61b6f2b241f2bbf45d156bd8c621e40bb2caed4d6967281a3`.
Raw evidence: `artifacts/parakeet-pointwise-tail-app-amd-20260927`.
