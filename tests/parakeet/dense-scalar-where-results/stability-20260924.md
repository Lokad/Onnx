# Where repeatability with identical release binaries

**Control rejected.** Both roles use selected Core `672e5f30` / Data `065b7a7f`; no optimization candidate is measured.

All correctness, identity and resource checks pass. 10 of 225 repeatability controls fail; 22 of 220 cases exceed the symmetric five-percent middle/outer-pair bound.

| Sum of case means | Outer pair ms | Middle pair ms | Middle / outer | Pair bound passes |
|---|---:|---:|---:|---|
| all220 | 55.938095 | 55.700012 | 0.995744 | True |
| target6 | 9.345856 | 9.275193 | 0.992439 | True |
| other_uniform42 | 9.279415 | 9.266305 | 0.998587 | True |
| fallback74 | 27.976644 | 27.860717 | 0.995856 | True |
| added98 | 9.336179 | 9.297797 | 0.995889 | True |

The large-nonscalar-false case takes 22.230238, 14.102600, 14.389612 and 23.426960 microseconds across the four identical-binary processes. Its middle/outer ratio is 0.624046, a false 37.6% improvement. All three measured 60-sample blocks per process retain the same broad level; one isolated spike cannot explain that case. This does not identify compilation, allocation/layout or another runtime mechanism as the cause. The previous vector-last problem is absent at its historical scale in this distinct run (four-process max/min 1.0762), without overturning the earlier rejection.

All four fresh CPU2 processes use ordinary .NET 10.0.8 with the same consumer and product hashes. Each prepares all 220 valid cases, then completes 600 warmup samples for every case before starting 180 measured samples per case. Batches and the complete CPUExecutionProvider.Where boundary are fixed. The original 122 cases retain their order and partitions; 98 added cases cover the new numerical boundaries and broadcast runs.

All inputs, exact result bits, OpResult metadata, full input stores, distinct outputs and held-output ownership pass. All clocks, including warmups and pauses, remain retained. Scores average every measured clock with equal process weights. The thirteen 60-sample blocks per case are descriptive only; none replaces the frozen score.

All seven jobs, 2,259 resource observations, 880 setups and 686,400 clocks pass. There are 232,929,840 complete public calls, including 53,753,040 measured calls. Peak RSS is 416,915,456 bytes.

Consumer SHA-256: `19f25657ce5abf2af66d6d2e499a8ead0a002c49b3ba5f5c76a03b61ac014dd2`. Closure: `882ac1267e5ef80fc7b8bf2b01f3252f5b72cba8fb2fc1da2c783fa9e7e27986`. Tools were frozen at `2a0b9e7d`; owner 897724 / birth 1790217849.76 and all descendants are terminal.

[Every case](stability-cases-20260924.csv), [all phase blocks](stability-phase-blocks-20260924.csv), and [exact scoring plus raw-file identities](stability-observations-20260924.json) are published. All 686,400 raw clocks remain in `artifacts/parakeet-dense-scalar-where-stability-amd-20260924/clocks.csv` and the verified collection archive. Keeping one retained clock export avoids another large tracked duplicate.

This control does not admit the measurement protocol for candidate scoring. Preserve the failure and investigate it before making a speed claim; do not retry the unchanged control or use favorable blocks. The candidate remains numerically qualified and unmeasured.

Root product and BENCHMARK.md remain the fully qualified selected release.
