# Parakeet component qualification

This opt-in lane adapts the encoder/decoder cases and independently carried
state checks from voice branch `2b1138fe6f5d085e3749f6867d1603b1131ff029`.
It compares **every output value**, including exact integer lengths, against
native ONNX Runtime 1.29.0. It does not require native ORT in the managed replay
or add a dependency to the shipped library.

The pinned export is `istupakov/parakeet-tdt-0.6b-v3-onnx` revision
`8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce`. `assets.json` identifies the FP32
encoder graph, its 2.4 GB external weights, and the decoder/joint graph. Stage
regular local files explicitly; inference never downloads them:

```powershell
hf download istupakov/parakeet-tdt-0.6b-v3-onnx encoder-model.onnx encoder-model.onnx.data decoder_joint-model.onnx --revision 8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce --local-dir models/parakeet-tdt-0.6b-v3
python -m pip install -r tests/parakeet/requirements.txt
pwsh -File eng/test-parakeet.ps1 -Output artifacts/parakeet-new
```

The wrapper builds the replay, generates new native references and executes the
managed model. Existing output directories are refused. Missing assets or failed
comparisons fail the command; ordinary unit tests do not download or require
these large models. Individual steps are:

```powershell
python tests/parakeet/generate_reference.py --models models/parakeet-tdt-0.6b-v3 --output artifacts/parakeet-new/reference
dotnet build tests/parakeet/ParakeetReplay.csproj -c Release --tl:off --nologo -v minimal
dotnet tests/parakeet/bin/Release/net10.0/ParakeetReplay.dll models/parakeet-tdt-0.6b-v3 artifacts/parakeet-new/reference/manifest.json artifacts/parakeet-new/managed.json
```

The encoder receives float `[batch,128,frames]` features and int64 lengths. Cases
cover 64, 65, 128 and 256 frames, plus batch two with different valid lengths.
The decoder receives float `[batch,1024,frames]`, int32 target IDs and lengths,
and two float `[2,batch,640]` states. Cases cover single-frame/single-token and
multiple-frame/multiple-token requests, batch two, and multiple recurrent steps.
Every engine advances its own states. The fixture manifest cannot substitute
saved native states for managed recurrent outputs.

Four invalid-input calls must fail in both engines, followed by successful,
bit-identical repeats of their own initial requests. Every input and retained
output is hashed after later execution and Reset. All 60 successful output
comparisons require matching shapes/dtypes, exact integers, and finite floats
with `abs(actual-reference)/max(1,abs(reference)) <= 1e-4`.

References bind all NPY files, model assets, native versions/settings and the
generator source embedded in the replay. Source identities additionally normalize
CRLF to LF so Windows checkouts and source archives bind the same text. Raw file
hashes are retained separately. Asset byte hashes are never normalized. Every
external tensor reference must stay in the pinned model directory and within its
sidecar's bounds.

Managed execution uses private `ExecutionOptions.Memory` contexts and packed
weight limits of 256 MiB for the encoder and 64 MiB for the decoder. It verifies
that native ORT is absent and records actual assembly hashes, settings, complete
errors, allocation observations and process peak working set. The reported
load/execution durations are diagnostics, not a controlled performance comparison.
Complete managed outputs are retained beside the result in `<result.json>.tensors`
as little-endian raw arrays; the result records their shapes, dtypes and hashes
for independent comparison with the native NPY files.

This establishes component behavior on deterministic synthetic inputs. The audio
frontend has a separate [full-feature replay](frontend/README.md). Vocabulary/text decoding, token/duration advancement,
long recordings and complete Parakeet transcription remain separate work. Passing
this lane must not be reported as end-to-end ASR support or qualification of the
INT8 exports.
