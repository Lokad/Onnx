# Reused-context Whisper comparison: resource limit — 2026-09-20

The reused-context attempt also stopped at the unchanged **8 GiB process-group
RSS limit**, at **8,791,007,232 bytes** after 240.10 seconds. All 472 samples are
retained; time and available-memory guards passed. The minimum available memory
was 2,053,832,704 bytes. Only the owned process group was stopped; the supervisor
and both observed children are confirmed absent. No native worker started.

Twelve of 84 planned encoder outputs completed: six original managed baseline
bridges and six managed-encoder/native-feature outputs. All six bridges match
saved output bytes exactly. Every completed full array, input/shape/hash/gate
record, held output and allocation observation was independently checked. This
prefix does not establish the full twenty-recording input/engine decomposition.

One Memory GraphExecution was retained and Reset around every call. The changed
lifecycle was insufficient to complete the experiment within its bound. At the
second completed call, managed heap estimate was 3,467,724,816 bytes; at the
last it was 8,329,352,592 bytes. Collection counts remained [7, 7, 6], with most
recent GC index 7, throughout that interval. Allocated-byte counters rose by
approximately the same amount. These observations show allocation accumulation
since the last completed collection; they do not establish which objects remain
reachable, an unbounded leak, or the behavior of a complete public transcription.

The completed calls recorded 4,519,680,000 newly allocated pool payload bytes and
213,899,520,000 bytes served from pool reuse. Those are cumulative counters, not
peak or retained memory. The first fresh-context attempt remains a separately
closed failure. Neither attempt licenses a relaxed memory or numerical limit.

The next experiment will use finite worker lifetimes for each recording, retaining
the complete corpus and all baseline bridges. Process isolation changes the
execution lifetime, so it supplies numerical localization rather than a memory
or performance qualification of the public transcriber.

Original producer source: `2b67929`. Manifest SHA-256:
`90ac9ced9d1a094509f785d59bcc04b9c42ad9bfa6ac6d36d99f55178460491d`.
All arrays, resource samples, actual process births and exact source snapshots
are retained in `artifacts/whisper-input-cross-reuse-20260920`.
