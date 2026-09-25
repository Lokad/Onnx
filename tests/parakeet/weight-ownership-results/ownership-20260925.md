# Parakeet: original-weight ownership

The ownership prediction passed. Removing two sets of references made all 96
original feed-forward weight arrays collectible while all 37 existing packed
copies retained their identities and bytes. Live managed memory fell by
1,610,542,840 bytes, approximately 1.5 GiB. This establishes the ownership
mechanism; a replacement representation has not yet been tested in inference.

| Disposable transcriber state | Original arrays alive | Live managed bytes | Process RSS bytes |
|---|---:|---:|---:|
| Before removal | 96 | 2,899,511,536 | 3,190,206,464 |
| Original initializer bindings removed | 9 | 1,439,961,656 | 3,132,768,256 |
| Corresponding packing-map entries also removed | 0 | 1,288,968,696 | 3,091,992,576 |

Tensor weak references follow the same 96 → 9 → 0 pattern. The nine remaining
originals in the middle snapshot were exactly those referenced by packing-map
keys and records. The transcriber, encoder and two execution contexts remained
alive throughout; both contexts shared the encoder dictionaries. All other
initializer identities remained unchanged. The final map retained its 28
unrelated entries. Forced collections occurred only in this unscored experiment.
The much smaller RSS reduction illustrates why managed liveness must be reported
separately from memory returned to the operating system.

Before any removal, the same transcriber completed corpus clip
`1188-133604-0002`: 287,360 PCM samples, 225 encoder frames and 101 decoder calls.
Text, token IDs, frame indices, durations and termination matched the retained
reference exactly. The input and held public result remained unchanged.
**No inference ran after removing the bindings.** The experiment left a
deliberately incomplete graph; it does not establish that deletion alone is a
valid optimization or that the remaining packed copies cover all 96 weights.

The [preceding cost diagnosis](../feed-forward-cost-results/diagnosis-20260925.md)
measured 3.149526 seconds of repeated packing against a 3.855585-second difference
from the earlier ORT feed-forward profile. At the installed ORT source revision,
`SessionState` removes an original initializer when every consumer has prepared
it. Lokad's current representation retains the original tensor and array as
well as its prepared copy. This experiment confirms those managed roots rather
than inferring collectibility from the graph's consumer count.

The next bounded prototype is a logical weight tensor backed by one packed
array. First prove exact logical indexing, independent dense reconstruction,
and scalar/unsupported-instruction fallback using the existing multiplication
routines. Admit only single-consumer, privately owned constants. Keep the
ordinary mutable graph path intact; do not increase its clone budget. The
observed short-wide path and its odd-row remainder must retain their arithmetic
order. Dispatch, alias tracking and preparation invalidation must understand the
representation before any model trial. A successful layout probe would still
need full numerical, memory and application qualification; no saving is promised.

The probe reused Core `49c3a958` and Data `a893952f` without rebuilding either.
Its corrected consumer built with zero warnings and errors. The original
consumer build failed CS0411 and remains retained; the correction adds only the
explicit generic type argument to `MemoryMarshal.TryGetArray<float>`.
Capture completed in 8.578 seconds with 17 resource samples and sampled peak RSS
4,020,441,088 bytes. This elapsed time includes model loading and the ownership
experiment and is not a transcription benchmark.

[Audited snapshots, public result and source identities](ownership-20260925.json)
are published by [publish.py](publish.py). Closure:
`abe97a41fbb60648a6ee55968cd5e36c861ac908603336bd2f465379cbb1f3cd`.
The isolated Core still lacks release admission because two e5 repeatability
controls failed. Product source and `BENCHMARK.md` remain unchanged.
