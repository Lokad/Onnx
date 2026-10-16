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

The prepared graph retains projected token states. Host processing applies a
masked FP32 mean, keeps the requested leading dimensions, and normalizes.
The preparation tool preserves the original files and copies the external
weight sidecar into a separate directory. It expands RMS/skip normalization
and rotary embedding without allowing arbitrary custom operator domains.

The native and source references were established on the dedicated AMD VM on
2026-10-16: all 14 cases pass, including Unicode, padding, empty text, and
same-engine repeatability. The source/export maximum vector-coordinate error
is 2.39e-7; the prepared/original maximum component allowance ratio is 0.301.
Managed scalar, SIMD, and default execution each pass the same 14 cases at all
four dimensions; the largest observed vector-coordinate error is 5.67e-7.
This is frozen-tensor correctness evidence, with inputs up to 370 tokens,
not an end-to-end tokenizer/API or performance claim. Exact Gemma tokenization
is owned by Lokad.Tokenizers; the current runner consumes frozen Python tensors.
