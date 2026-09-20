# Whisper crossed-input attempt: resource limit — 2026-09-20

The first encoder-only attempt stopped at its fixed **8 GiB process-group RSS
limit**. The triggering sample was **8,797,216,768 bytes** at 177.83 seconds.
All 350 resource samples are retained. Time and available-memory guards passed;
the RSS guard stopped only the owned process group. The supervisor and both
observed children are confirmed absent. No native worker started.

Nine of the planned 84 encoder outputs completed: five managed baseline bridges
and four managed-encoder/native-feature outputs. All five bridges reproduce the
saved baseline bytes exactly. Every completed output, shape, input hash, numerical
count and held-output record was independently checked. This incomplete prefix
does **not** establish the planned full-corpus input/engine decomposition.

The probe created a fresh execution context for each encoder call. Source review
confirms that the original public transcriber also creates fresh contexts per
request. Explicit contexts have independent released-buffer caches, but this run
does not distinguish retained caches, temporary allocations and GC behavior as
causes of the rising RSS. It is not evidence of an unbounded production leak.

The 8 GiB guard remains unchanged and this failed attempt will not be replayed
unchanged. A future bounded diagnostic must declare its execution-context or
process-lifetime change before inference and require all baseline bridges again.
No product code, numerical threshold, benchmark ratio or qualification changes.

Original tool source is `9571a6a`. Manifest SHA-256:
`0818f1a10c27ea525d63b87aa0e66f792a40d6da9be0988d6bac11fe45b05e2e`.
The complete failure receipt, full completed arrays, resource samples and exact
source snapshots are retained in `artifacts/whisper-input-cross-20260920`.
