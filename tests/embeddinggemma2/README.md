# EmbeddingGemma 2 conformance

This lane compares the pinned Google source model, the community ONNX export,
a prepared text-only graph, and Lokad.Onnx. Python never loads managed code.
The model assets and generated arrays are gitignored; `assets.json` pins the
download revisions and SHA-256 hashes.

Create a Python environment with `python -m venv artifacts/embeddinggemma2-venv`
and install `python/requirements.txt`. Use that environment's Python for:

    python tests/embeddinggemma2/python/download_assets.py
    pwsh -NoProfile -File eng/test-embeddinggemma2.ps1 -Python <environment-python>

The lane requires complete local weights, verifies their hashes, and compares
all hidden states and 128/256/512/768-dimensional vectors. It also compares
repeated managed outputs byte for byte. Missing assets or missing cases fail.
The scalar mode is intentionally slow. This correctness lane fixes JIT tiering;
it supplies no performance claims. Performance qualification uses the existing
isolated-process benchmark protocol under ordinary runtime settings.

The source-model reference uses a separate environment installed from
`python/requirements-source.txt`, with CPU PyTorch. In that environment run:

    python tests/embeddinggemma2/python/download_assets.py --source
    python tests/embeddinggemma2/python/source_reference.py --source-dir models/embeddinggemma-2-source --reference artifacts/embeddinggemma2/reference --cases tests/embeddinggemma2/cases.json --output artifacts/embeddinggemma2/source-reference

The reference records exact tokenizer IDs/masks and binary `.npy` outputs.
`--tokens-only` on `python/reference.py` needs only the tokenizer JSON.
Operator-rewrite tests need no model weights:

    python -m pytest tests/embeddinggemma2/python/test_preparation.py -q

The prepared directory contains `model.onnx`, `model.onnx_data`, `tokenizer.json`,
and a hashed `preparation.json`. Once prepared, text inference is entirely local:

    lonnx embed models/embeddinggemma-2/prepared-fp32 --query "Which planet is known as the Red Planet?" --dimensions 768
    lonnx embed models/embeddinggemma-2/prepared-fp32 --documents input.jsonl --batch-size 3 --dimensions 256

JSONL documents have a string/number `id`, string `text`, and optional string/null
`title`. Outputs retain the ID and contain `embedding`, `dimensions`, `tokenCount`,
`spaceId`, `model`, and `precision`. The default limit is 512 tokens including
the task prompt and BOS/EOS; overflow fails without truncation. The 128/256/512/768
outputs truncate and renormalize. Batches are bounded by both row count and padded
token count. Larger token limits need separate numerical/resource qualification.

The source-level `EmbeddingGemma2Embedder` API in nonpackable Lokad.Onnx.Data
loads the prepared assets once, serializes calls, checks cancellation between
batches, and returns independently owned arrays. Query/document prompt formatting
is in Onnx; BPE is supplied by qualified Lokad.Tokenizers 0.2.0. Restore that pinned
package with `pwsh -NoProfile -File eng/restore-tokenizers.ps1` before building.

The full lane includes CLI text-to-vector checks against the Python references
at all four dimensions, padded JSONL batches, token counts, space identity, and
explicit invalid-input rejection. `-FrozenInputsOnly` runs only frozen graph conformance;
it does not qualify the public text pipeline.

The API runner also compares exact integrated inputs and vectors, right padding,
reused contexts, same-engine byte repeatability, caller-owned arrays, concurrent
callers, cancellation, and recovery after rejected requests:

    dotnet tests/Lokad.Onnx.EmbeddingGemma2Runner/bin/Release/net10.0/Lokad.Onnx.EmbeddingGemma2Runner.dll api models/embeddinggemma-2/prepared-fp32 tests/embeddinggemma2/cases.json artifacts/embeddinggemma2/reference artifacts/embeddinggemma2/api
    python tests/embeddinggemma2/python/verify_api.py --cases tests/embeddinggemma2/cases.json --reference artifacts/embeddinggemma2/reference --actual artifacts/embeddinggemma2/api

For byte-repeatability qualification, disable tiered compilation/PGO as the full
PowerShell lane does. This is a correctness setting; benchmarking retains its
separately calibrated runtime settings.

The prepared graph retains projected token states. Host processing applies a
masked FP32 mean, keeps the requested leading dimensions, and normalizes.
The preparation tool preserves the original files and copies the external
weight sidecar into a separate directory. It expands RMS/skip normalization
and rotary embedding without allowing arbitrary custom operator domains.

The native and source references were established on the dedicated AMD VM on
2026-10-16: all 15 cases pass, including Unicode, padding, empty text, exactly
512 tokens, and
same-engine repeatability. The source/export maximum vector-coordinate error
is 2.39e-7; the prepared/original maximum component allowance ratio is 0.301.
Managed scalar, SIMD, and default execution each pass the same 15 cases at all
four dimensions; the largest observed vector-coordinate error is 5.67e-7.
Each mode also repeats all 75 hidden-state/vector arrays byte for byte. The
integrated API matches exact input IDs/masks for 14 text cases and passes 52
single/batched vector comparisons. The CLI passes 28 vector comparisons and
rejects 513 tokens. The maximum integrated vector-coordinate error is below
5.25e-7. These are correctness results; no performance qualification is implied.
Exact Gemma tokenization is owned by Lokad.Tokenizers.
