# Parakeet: measure weight preparation before choosing the next change

The next experiment asks one question: **how much of the remaining feed-forward
projection cost is weight preparation, and how much is multiplication?** No new
optimization is selected until the complete cost split and observer controls pass.

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
and the failed audit are retained. Continuation runs only the unfinished stages
and unstarted markers roles, using the exact same binaries and workload with
separate 2 GiB trace allowances. All numerical, repeatability, observer-effect,
CPU, memory and elapsed-time limits remain unchanged.

The split is not yet available. If preparation is small, stop pursuing packing
changes and inspect the generated multiplication loop for the actual route.
If preparation is substantial, first establish which retention mechanism can
remove it within the existing memory budget. Either outcome must yield one
testable prediction, with a measured affected cost and a clear rejection result.
