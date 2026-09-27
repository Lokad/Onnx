# Unmapped calls recover over several repetitions

The first-call-only hypothesis is **not supported**. The diagnostic reproduces
the unmapped slowdown (1.963929 times current), but only 49.4187% of positive
per-position excess occurs in call one, below the prospectively fixed 80%.
Calls two through four are also slower. Calls five through seven approach current.
The packed-row screen remains rejected; no release or application gain is claimed.

| Position in seven-call unmapped batch | Current ms | Candidate ms | Candidate/current |
|---|---:|---:|---:|
| 1 | 0.407736 | 1.766378 | 4.332157 |
| 2 | 0.406260 | 1.396424 | 3.437265 |
| 3 | 0.413633 | 0.751232 | 1.816178 |
| 4 | 0.405331 | 0.453147 | 1.117967 |
| 5 | 0.404835 | 0.409618 | 1.011815 |
| 6 | 0.402672 | 0.412913 | 1.025431 |
| 7 | 0.407340 | 0.403525 | 0.990634 |

Both candidate processes show this recurring recovery throughout the original
warmup and measurement periods. This is inconsistent with a uniform twofold
slowdown of every fallback call. It does not identify cache misses, active JIT
versions or individual instructions. All thirteen consecutive sixty-round blocks
and every individual interval remain available; no clock is removed.

The VM's post-capture sysfs report exposes a 32 MiB shared L3 cache for CPUs 0–3,
1 MiB L2, 32 KiB L1 data, and 64-byte lines. Each original or packed weight array
is 20,986,880 bytes; together they total 41,973,760 bytes, exceeding reported L3
capacity. In the fixed schedule, current mapped/unmapped calls reuse one array;
candidate transitions from packed to original. This makes cache competition a
plausible explanation for the repeated recovery. Capacity and timing correlation
are not hardware-counter proof, and virtualized topology is reported topology.

The narrow controls also retain process variation. This time the current product
fails repeatability: prediction max/min 1.132341, encoder 1.117452. Their paired
candidate/current ratios are 0.931231 and 0.949017. The earlier candidate-side
failures remain failed. The role reversal argues against treating the original
paired width means as proof of a fixed candidate-only slowdown, but does not
establish fallback performance equivalence or identify the cause of that variation.

Only the common consumer changes: b605e347 adds two preallocated timestamp arrays
and individual clocks around the unmapped calls. Core remains current 65f15a41
and candidate af19b3b4. Original six cases, products, order, 600 warmup rounds,
180 measured rounds and seven/102 calls per batch remain fixed. All 723,840 calls,
18,720 batch clocks and 21,840 additional intervals are retained. Exact checks
cover 2,784 outputs; all input/weight/output hashes and ownership checks pass.
The outer-minus-inner timing gap is below 0.9 microseconds per batch mean in each
process. These instrumented clocks are diagnostic, never replacement admission.

All four ordinary CPU2 processes exit zero. Build audit reports no warnings.
Closure 966941d1 binds raw clocks, products, resources and every completed owner.
Cache receipt 73bc117d was collected after confirming all owners terminal.
Reproduce the JSON from the retained inputs, without execution:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/decoder-packed-row-results/unmapped_calls.py

The next decision concerns the unchanged candidate in the actual Parakeet
sequence. PLAN proceeds to full-model numerical/decision/ownership validation,
then an independent complete twenty-clip application comparison with the already
declared 1% improvement and 5% per-clip regression limits. This is a new workload
decision after diagnosis, not retrospective admission of either failed hypothesis.
Broader supported-model checks, actual build/package qualification and explicit
disclosure of the mixed-layout control limitation remain required for release.
No further kernel, warmup, tile, flag or cache-conditioning search is selected.
