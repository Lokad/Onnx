# Why the generic Where insertion needs a different boundary

The untimed probe captures a concrete change in the fallback's compiled structure.
Both products emit a complete float Tier1 entry, as well as instrumented Tier0 and
optimized loops entered through on-stack replacement (OSR). The hypothesis that
the candidate never produces a complete optimized entry is unsupported here.

| Complete float `Where` entry | Selected | Candidate V3 |
|---|---:|---:|
| Native bytes | 2,893 | 2,426 |
| Profiled functions incorporated into the caller | 23 | 17 |
| Calls to `BroadcastShape` outside the caller | 0 | 3 |
| Calls to the uniform helper | 0 | 1 |
| Integer divisions | 4 | 4 |

The selected entry incorporates the three shape-broadcast operations. V3 retains
three calls to `BroadcastShape`. Both report synthesized profile information;
their recorded call counts differ. The consumer's generated bodies always call
the public `Tensor<T>.Where` boundary once per loop iteration, across all nine
dtypes. They never call the new helper directly. The helper remains a 1,277-byte
FullOpts body with byte-zero/nonzero tests and independent fill/copy output.

All 65 bodies were retained with complete endings and resolved branch labels:
33 selected and 32 candidate. The selected consumer has 20 emitted bodies and
the candidate 18; both cover all nine dtypes. These different compilation histories
are evidence from separate untimed processes. They do **not** identify the tiers
used in the rejected performance screen, attribute its full regression magnitude,
or change that rejection.

The probe performs 17,640,480 public calls across two workers, using all 122 valid
qualified cases in their frozen order and batch counts, with 120 repetitions.
Its execution loop contains no stopwatch or clock collector. Exact output bits,
shapes, full input stores/guards, held outputs and ownership pass. All five jobs
and 134 resource observations pass, with peak RSS 332,636,160 bytes. Owner889563/
birth1790211402.22 and every descendant are terminal, code0.

The next distinct hypothesis moves the specialization to the execution provider's
existing float `Where` dispatch and preserves the entire generic tensor method.
It needs its own source/build/numerical/generated-code/component qualification.
Component timing must include the complete provider call and returned `OpResult`;
the old public-tensor screen remains rejected and supplies no admitted gain.
No application run or integration is authorized by this diagnostic result alone.

Frozen tools:485f9416,tests/parakeet/scalar-where-fallback-codegen.
Closure:1cf6a1241cb1501fdf24e7dc049b19181353f3d80c9dfacd71729ece384ffd41.
Consumer:b106e54f32a18356d557ebad08fca0195bbddb079acb13401a4350028075a285.
[Complete generated-code review](fallback-codegen-review-20260924.json):
4f8b356456d9929db8a69532dfbceff2019044f016d8a111e921995fd58af1b5.
Local bodies:artifacts/parakeet-scalar-where-fallback-codegen-review-20260924.
Full logs/results:artifacts/parakeet-scalar-where-fallback-codegen-amd-20260924.
Selected release and BENCHMARK.md remain unchanged.
