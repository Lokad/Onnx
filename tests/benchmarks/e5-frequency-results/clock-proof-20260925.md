# Frequency-counter clock alignment proof

**The corrected no-model proof passes.** Its perf epoch is bounded to
19.197999 ms, below the prospective 20 ms limit.
All 1,000 calls to the installed .NET timestamp function lie inside their
Linux monotonic-clock brackets. A complete counter interval lies within the
known arithmetic workload and reports positive APERF, MPERF and TSC counts.
All three raw intervals remain retained. No model or product build ran.

The first attempt rejected an incorrect forced-output assumption. Although
enable is acknowledged, the handler returns zero and the caller's positive-result
branch does not print an immediate interval. The proof refused to treat the
next periodic reading as a 20 ms handshake. The second attempt completed the
arithmetic workload but rejected the next acknowledgment: the reader consumed
four bytes while perf sends five, including a NUL terminator. Both failed runs
and all their source/data remain retained. See the
[control handler](https://raw.githubusercontent.com/torvalds/linux/v6.17/tools/perf/util/evlist.c)
and [interval dispatch](https://raw.githubusercontent.com/torvalds/linux/v6.17/tools/perf/builtin-stat.c).

The successful component reads the exact acknowledgment. It bounds the initial
epoch between launch and enable acknowledgment. At exit, ping establishes that
earlier interval output is complete; the proof snapshots it, stops the bounded
helper and requires new interval output before perf exits. Subtracting that
reading's relative timestamp gives a second epoch bound. The two bounds overlap.
The final handshake and shared uncertainty both remain within 20 ms.

Ten local checks pass, including two successive fragmented acknowledgments,
missing/duplicate/malformed counters and incompatible epoch bounds. The actual
proof passes 75 resource observations with peak owned RSS
43,630,592 bytes. All supervisor, perf and workload owners are terminal.
No system settings, warmup, scoring limit or release product changed.

This proves the observation mechanism, not that frequency explains e5 timings.
The next workload is two identical original M73 candidate e5-512tok processes,
each preserving all 780 calls and the original numerical/ownership assertions.
Counter intervals and events must all be retained; no diagnostic clock replaces
the original failed release comparison.

[Closures, bounds and retained failures](clock-proof-20260925.json),
[every proof interval](proof-intervals-20260925.csv).

Successful closure: `8f63ad0396ef8fdb762afa2cf88bece14f0d6e9b49c4b033a1ad5906a323778f`.
