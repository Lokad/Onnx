# Verify the exact attention preparation scope on the real model

Use the unchanged candidate Core ee5218db and qualified Data 1ba343fd, after
focused-contract closure a293403a. Build only the existing census consumer.
Reversible source edits adapt its expected counts and payload sizes to include
the 120 square attention matrices alongside all 96 feed-forward matrices.
Keep its public-transcription, identity, logical-value, map, resource and
idempotence checks. No kernel, product source, GC policy or model changes.

Derive expected logical hashes directly from external-data ranges in the pinned
ONNX export already on disk. Verify complete model/external-data digests, float32
types, exact initializer names and shapes against the fresh projection mapping.
Require all 96 previously captured feed-forward hashes to match this independent
extraction. Of 216 selected weights, 179 must be independently owned: the original
87 feed-forward weights plus exactly 92 attention weights. The other 37 retain
their existing dense sources and cache entries. Every logical hash must match the
original export, and all 37 packed-map hashes and the 256 MiB budget must remain
unchanged. The total owned payload is 1,845,493,760 bytes.

Run one census and complete longest-clip transcription in each normal and
disabled-AVX512 mode. Check immutable PCM, exact public tokens/frames/durations,
unchanged initializers and map identities, shared prepared contexts, and exact
before/after payload hashes. No forced collection or application timing score.
This is followed by the original complete numerical/public corpus and the
independent application comparison before admission.

Reuse the original consumer-only worker and auditor. Build jobs require 2 GiB
available RAM / 1 GiB tmpfs, 3 GiB RSS and 180 seconds. Each census requires
11 GiB RAM / 2 GiB tmpfs, 12 GiB RSS and 900 seconds. Preserve 1 GiB free
RAM/tmpfs, 128 MiB output, half-second resource samples, CPU2 compute and CPU0
monitor. SDK10.0.204/runtime10.0.8 and the existing offline feed are unchanged.

From repository root, prefix with C:/Python313/python.exe -X utf8 -B:

    tests/parakeet/attention-owned-census/run.py prepare
    tests/parakeet/attention-owned-census/run.py stage
    tests/parakeet/attention-owned-census/run.py launch build
    tests/parakeet/attention-owned-census/run.py observe build
    tests/parakeet/attention-owned-census/run.py collect build
    tests/parakeet/attention-owned-census/review.py build
    tests/parakeet/attention-owned-census/run.py launch capture
    tests/parakeet/attention-owned-census/run.py observe capture
    tests/parakeet/attention-owned-census/run.py collect capture
    tests/parakeet/attention-owned-census/review.py capture

Freeze tools before preparation; mutations write once. Observe is read-only.
Collect only terminal owners. Preserve failures and diagnose retained evidence
before recovery. Preparation reads existing local model files; it runs no local
inference and downloads no model. Do not publish a performance gain from this lane.
