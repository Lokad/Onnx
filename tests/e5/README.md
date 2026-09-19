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
required; without x86 FMA the intrinsics run records an unsupported-marker
artifact and its three conformance tests skip instead of failing), then runs
`tests/e5/python/test_e5_conformance.py`, which checks
exact tokenizer inputs, full hidden states, pooled embeddings, and the
semantic ranking margin (>= 0.10, measured 0.18).

## Current AMD performance

The [fresh public-options / ORT comparison](public-ort-20260919.md) records the
qualified production defaults in ninety processes and 5,940 measured calls.
Default/ORT means are 1.0212, 1.1059, 1.0919 and 1.1198 for 8, 30,
30 padded to 128, and 128 tokens. Explicit Memory ratios are 0.9747, 1.0808,
1.0751 and 1.0746. Thus three primary cases still exceed the 5% target in
these descriptive observations. Complete output/ownership checks pass;
historical statistical qualification remains unresolved.

The [earlier September 19 comparison](comparison-20260919.md) records the frozen
release, current experimental configuration and native ORT on the designated
AMD host. Current mean latency is 38–62% below release on the primary cases.
Current/ORT mean ratios are 0.9472, 1.0831, 1.0740 and 1.0731 for 8, 30,
30 padded to 128, and 128 tokens. These are descriptive observations with
process variation retained; they do not certify parity or change defaults.

The [CPU defaults qualification](defaults-20260919.md) records full local/AMD
regressions, audio decisions, resource observations and package consumption for
the nine enabled mechanisms. It separates those checks from measurement of the
actual public Default and Memory configurations. The completed
[public-options comparison](public-options-20260919.md) observes 17–41% lower
primary mean latency with the nine mechanisms enabled on the same current core,
with complete outputs unchanged.

## Tokenizer rule

The Python oracle reproduces the XLM-R fairseq framing independently with the
`sentencepiece` library: bos=0, piece ids shifted by +1, eos=2, first 510
pieces kept. Validated ID-for-ID against the Hugging Face reference tokenizer.

## Experimental packed MatMul remainders

`LOKAD_ONNX_PACKED_AVX512_NARROW=1` enables AVX-512 two- and three-row
remainders for prepared MatMul when `LOKAD_ONNX_PACKED_AVX512_ROWS=1` is also
set before process startup. The row-sharing route is enabled by default; the
narrow-remainder experiment remains off. The route keeps
the existing 32-column packing, 12/8-row bulk tiles, accumulation order and
destination behavior. Unsupported hardware, column tails and shapes without
a narrow remainder use the existing routes. The separate packed-panel
experiment takes precedence when enabled.

Run the direct arithmetic, offset, accumulation and ownership checks with:

    dotnet test tests/Lokad.Onnx.Backend.Tests -c Release --tl:off --nologo -v minimal --filter FullyQualifiedName~PackedAvx512NarrowTests

The arithmetic tests require AVX-512F and FMA; refusal tests also run on other
hosts. At source `8e93aa72388b404668434aaacfaf7a5ab1e7d42e`, actual AMD testing
passed 45 active new tests and the existing affected operator/lifetime suite.
Twenty fresh full-model workers passed native output agreement before and
after measurement, with maximum scaled error `1.501e-6`.

The September 19 diagnostic retained 660 measured calls and 26,706 conditioning
calls with normal tiering and GC. The thirty-token mean was 1.55% lower with
the switch enabled, while unchanged cases also moved. This is insufficient
to establish a model speedup or promote the switch. Local evidence is retained
under `artifacts/packed-narrow-product-v2-20260919`, including the failed first
supervisor attempt under `artifacts/packed-narrow-product-20260919`.
The successful result archive SHA-256 is
`7b216fcf139f008aa94fc9ead98b1b999161731162cf1075ae6b2b40037190bd`.
See [the isolated evidence protocol](../../eng/isolated-evidence.md) for the
measurement boundaries and separate scoring requirements.

## Experimental exact BiasGelu scheduling

`LOKAD_ONNX_BIAS_GELU_INTERLEAVED=1`, together with
`LOKAD_ONNX_BIAS_GELU_INLINE=1`, interleaves four exact erf streams on qualified
AVX-512/FMA hosts. Other hardware and unsupported shapes use existing paths.
Inline BiasGelu is enabled by default; interleaving remains off. See the
[arithmetic proof and whole-model observations](interleaved-gelu-20260919.md)
for the precise geometry, tests, process variation and retained evidence.
