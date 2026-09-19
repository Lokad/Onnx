# Managed WeSpeaker embeddings

`Lokad.Onnx.Data` extracts a 256-value speaker representation from local normalized mono 16 kHz PCM:

```csharp
var model = new WeSpeakerEmbedder(
    "models/pyannote-embedding/embedding_encoder.onnx",
    "models/pyannote-embedding/projection.onnx");
WeSpeakerEmbedding embedding = model.Extract(
    monoSamples, 16000, ReadOnlySpan<float>.Empty, cancellationToken);
if (embedding.Status == WeSpeakerEmbeddingStatus.Completed)
{
    IReadOnlyList<float> vector = embedding.Values;
}
```

Inputs must contain 400 through 480000 finite samples in `[-1,1]` (at most 30 seconds). Empty weights select ordinary mean/sample-deviation pooling. A nonempty mask contains finite values in `[0,1]`, covers the whole recording uniformly, and may contain at most 480000 weights. Its length can differ from the model's frame count. The API uses pyannote's nearest-neighbor mapping `floor(frame * maskLength / encoderFrames)` and unbiased weighted statistics, followed by the affine projection. It does not normalize the final vector.

Fewer than two positive weights **after resizing** returns `InsufficientFrames` with an empty vector. `EncoderFrames` and `PositiveFrames` make that decision visible. A completed vector is not a speech-detection or speaker-reliability decision: silence with enough frames still produces a mathematical vector. Speaker-mask selection, overlap exclusion, clustering, speaker names and diarization timelines belong to the later pipeline.

Each result owns read-only storage. Requests on one instance serialize, use private execution contexts and preserve caller inputs. Cancellation is checked during preprocessing and between model calls; an in-flight model call completes before cancellation is observed. Invalid PCM or mask values are rejected even in an unused tail or an insufficient-data case. No model downloads, Python or native ONNX Runtime are used by the API.

## Local assets

Use `embedding_encoder.onnx`, `resnet_seg_1_weight.npy` and `resnet_seg_1_bias.npy` from the [WeSpeaker Community-1 split export](https://huggingface.co/welcomyou/pyannote-community-1-onnx-split/tree/cde44c2db938c8abb755853b9a87cb3179c47803), revision `cde44c2db938c8abb755853b9a87cb3179c47803`. The model card declares CC-BY-4.0; retain attribution to the export authors and pyannote Community-1. Preparation verifies the exact sidecar hashes, dimensions, dtype and finite values, then writes the missing projection as a standard ONNX graph:

```powershell
python eng/prepare-wespeaker-projection.py --model-directory models/pyannote-embedding --output models/pyannote-embedding/projection.onnx
```

This optional tool requires NumPy and ONNX, downloads nothing, refuses existing output files and writes a provenance JSON file. Runtime inference uses only the two ONNX files; the test fixture reader is not part of the product.

## Qualification

Ordinary tests use independent small statistics and tiny generated ONNX graphs, with no large assets or Python. They cover weighted/unweighted arithmetic, frame boundaries, nearest masks, malformed inputs/outputs, ownership, cancellation, recovery and concurrent requests.

The native lane covers 55 cases, each repeated twice: 52 returned vectors / 13,312 values and 58 explicit insufficient-frame results. All final-vector values pass `abs(managed-native)/max(1,abs(native)) <= 1e-4` locally, with maximum `6.86199e-6`. The native reference uses pinned pyannote source, Torch/Torchaudio 2.11.0 CPU and ORT 1.29.0; `pins.json` records identities. The separate [frontend qualification](../frontend/README.md) retains three failing values. Successful vectors do not erase that failure or establish full diarization, broad speaker accuracy or indefinite service memory behavior.

```powershell
python tests/pyannote/speaker/generate_reference.py --model-directory models/pyannote-embedding --reference-source external/pyannote-audio --frontend-reference artifacts/wespeaker-reference-new --output artifacts/speaker-reference-new
dotnet build tests/pyannote/speaker/SpeakerReplay.csproj -c Release --tl:off --nologo -v minimal
dotnet tests/pyannote/speaker/bin/Release/net10.0/SpeakerReplay.dll artifacts/speaker-reference-new models/pyannote-embedding/embedding_encoder.onnx artifacts/speaker-result-new.json
python tests/pyannote/speaker/audit.py --reference artifacts/speaker-reference-new --result artifacts/speaker-result-new.json --output artifacts/speaker-audit-new.json
```

First produce the frontend reference using its documented recipe. All output paths must be fresh. The speaker generator reexecutes the pinned native frontend, backbone, statistics and projection. Zero/one-frame decisions are the documented public API policy, not a claim that upstream returns the same status. The replay checks input/output preservation, native-runtime absence, malformed-request recovery and simultaneous requests. The independent auditor verifies all raw arrays, shapes, counts, digests and mask decisions.
