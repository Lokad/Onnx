# Parakeet: feed-forward cost diagnosis

**Repeated weight packing is the main measured lead:** it costs 3.150 seconds per
corpus, against a 3.856-second feed-forward difference from the earlier ORT profile.
The complete cost split passes all 63 repeatability controls and both observer
controls. Investigate prepared-weight ownership before another kernel variant.

| Complete feed-forward work | Seconds per corpus |
|---|---:|
| Multiplication, including retained-weight calls | 21.459574 |
| Repeated weight packing | 3.149526 |
| Odd-row remainder | 0.353764 |
| Output preparation | 0.051538 |
| Separate scale operations | 0.034575 |
| Other stages and measured transition time | 0.038489 |
| **Total Lokad** | **25.087467** |
| **Earlier ORT, matching complete groups** | **21.231882** |

The native profile predates this capture and observer overhead remains included.
The packing interval is about 82% of that diagnostic difference; it is not a
promised application saving. Weight reuse can also change cache behavior and
kernel dispatch, so subtraction alone cannot predict an optimization's result.

For the 4096×1024 family, Lokad's main multiplication interval is 10.627267 seconds
against ORT's earlier 10.636332-second complete fused family. Lokad additionally
spends 1.719864 seconds packing, 0.145618 seconds on odd-row remainders, and time
on output preparation and scale. This is a specific reason to prioritize weight
reuse over changing that multiplication loop.

The [installed ORT investigation](../ort-diagnosis-results/ort-kernels-20260924.md)
identified the sampled native GEMM routine by matching all 8,904 bytes against
the corresponding Microsoft source. The optimized graph, runtime input profiles
and matching source support reuse of prepared constant weights. This is stronger
than inferring behavior from the ONNX model alone, but it is not a per-node dump
of ORT's packed buffers or a per-node observation of its selected assembly leaf.

The [matched feed-forward review](../slice-dense-conversion-results/feed-forward-review-20260925.md)
covers all 96 weights and actual sequence lengths. Lokad repeatedly packs many
of these weights; its packing budget and row guards explain which calls do so.
The retained scratch requests total 30.249 GB per corpus. Those bytes do not
tell us how long packing takes. The earlier inclusive packing-boundary trial
gained only 0.730% and failed its original performance criterion, so repeating
that change is not justified.

The diagnostic uses the isolated slice-conversion candidate, whose Parakeet
application comparison passes at 60.852 seconds against ORT 39.794 seconds.
It remains unintegrated because two e5 repeatability controls failed. The
qualified product and `BENCHMARK.md` remain at Parakeet 64.706 / ORT 39.636 seconds.

The diagnostic's compiled review verifies unchanged arithmetic, original method
flags and public declarations. It reuses the consumer and adds boundaries around
scratch rental, weight packing, multiplication, scratch return, row remainder
and output preparation. Complete enclosing calls include the corresponding
scale operations. Zero new Data warnings remain; existing Core warnings are
recorded. The two earlier failed diagnostic builds remain retained.

The first capture completed 80 clock-only requests, then stopped after 33 stage
traces at its 512 MiB output limit. Convolution generated hundreds of thousands
of timing records per request; RAM and elapsed-time guards passed. All raw files
and the failed audit are retained. The continuation completed only the unfinished stages
and unstarted markers roles, using the exact same binaries and workload with
separate 2 GiB trace allowances. All numerical, repeatability, observer-effect,
CPU, memory and elapsed-time limits remain unchanged.

All three complete roles pass their public-result, input, ownership, resource and
accounting checks. Clock/stages/markers corpus times are 58.442227/60.452490/
60.649949 seconds. Stage overhead is 3.44%; added-marker overhead is 0.33%, both
within the original 5% limits. All 63 repeatability controls pass. The audit joins
240 complete requests, 7680 feed-forward calls, all 2856 encoder nodes and the
48 corresponding scale nodes. No profiler overhead is subtracted.

The follow-up source comparison identifies a concrete memory question. All 96
feed-forward constants have one graph consumer, as MatMul input 1, and occupy
1,610,612,736 bytes. The encoder's 268,435,456-byte packed-clone budget is full.
At the installed revision, ORT releases the original initializer after every
consumer has successfully prepared it. Lokad's `PackedMatMulWeight` holds the
source tensor and array as well as its packed clone; execution contexts share
those references. This explains why copying ORT's retention policy is more than
raising a cache limit. The graph-use census is not a managed heap dump, and it
does not prove that Lokad can discard those originals without breaking replacement,
fallback or public initializer contracts.

The next bounded investigation is therefore whether immutable graph-owned
weights can retain one prepared payload while preserving those contracts. Check
ownership and actual memory feasibility before selecting one implementation.
Do not repeat the rejected packing-boundary change or sweep cache sizes.

[Exact cost totals, controls, source objects and all 96 weight consumers](costs-20260925.json)
are generated by the [read-only publisher](publish.py). Joint closure:
`d7b79da277b3f6017be67478e67d52c753dc0c2ffdb6d0f31ee27a84f6e9bcab`.
All complete and failed raw captures remain local; only verified terminal VM
trace duplicates were retired. No product source or release benchmark changed.
