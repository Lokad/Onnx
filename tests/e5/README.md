# Native e5 conformance

Compares Lokad.Onnx against native Python ONNX Runtime on the exact local
`models/multilingual-e5-small` assets. Python is a test-only oracle; it never
loads .NET.

## Assets (git-ignored, hash-verified)

- `models/multilingual-e5-small/model.onnx`
  SHA-256 `CA456C06B3A9505DDFD9131408916DD79290368331E7D76BB621F1CBA6BC8665`
- `models/multilingual-e5-small/sentencepiece.bpe.model`
  SHA-256 `CFC8146ABE2A0488E9E2A0C56DE7952F7C11AB059ECA145A0A727AFCE0DB2865`
  (fetch once from the intfloat/multilingual-e5-small Hugging Face repo)

## Cases

`tests/e5/cases.json`: 8 stable cases (plain English, e5 query/passage pairs,
French accents, Japanese, punctuation plus tab and NBSP, 512-token truncation).

## Run

    .\eng\test-e5.ps1 -Python C:\Python313\python.exe -RequireIntrinsics

The script verifies hashes, builds `tests/Lokad.Onnx.E5Runner`, runs the
scalar, SIMD, and intrinsic modes twice each (byte-identical artifacts
required), then runs `tests/e5/python/test_e5_conformance.py`, which checks
exact tokenizer inputs, full hidden states, pooled embeddings, and the
semantic ranking margin (>= 0.10, measured 0.18).

## Tokenizer rule

The Python oracle reproduces the XLM-R fairseq framing independently with the
`sentencepiece` library: bos=0, piece ids shifted by +1, eos=2, first 510
pieces kept. Validated ID-for-ID against the Hugging Face reference tokenizer.
