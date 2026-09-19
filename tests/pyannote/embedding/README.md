# Pyannote embedding backbone qualification

This opt-in lane adapts the useful embedding cases from voice branch
`2b1138fe6f5d085e3749f6867d1603b1131ff029` and checks **every output value**.
It runs the existing managed core without native ORT in the replay process.
There are no shipped dependency or operator changes.

The pinned [split export](https://huggingface.co/welcomyou/pyannote-community-1-onnx-split/tree/cde44c2db938c8abb755853b9a87cb3179c47803)
derives from pyannote Community-1 and the WeSpeaker ResNet34 embedding model
(Hervé Bredin et al.; Hongji Wang et al.). The export declares CC-BY-4.0.
Stage `embedding_encoder.onnx` under `models/pyannote-embedding`; its exact hash
and source revision are in [assets.json](assets.json). No model download occurs
inside the generator or managed engine.

Input `fbank_features` is float32 `[batch,frames,80]`; output
`/resnet/pool/Reshape_output_0` is `[batch,2560,ceil(frames/8)]`. It contains frame
features, **not final speaker embeddings**. Weighted statistics pooling, a final
projection, speaker-mask validity and clustering remain separate work.

The managed PCM frontend and its explicit numerical qualification limits are
documented in [frontend/README.md](../frontend/README.md). This backbone lane
continues to use pinned native filterbanks so that component and propagated
pipeline errors remain distinguishable.

Eight cases cover 200/201/400/800 frames, batch two, and recorded English, French
and JFK audio. Native filterbank inputs come from the actual pinned
`compute_fbank` method in [pyannote.audio](https://github.com/pyannote/pyannote-audio/blob/b749285c5cdd4636b2edc7f766f1352c8dde9369/src/pyannote/audio/models/embedding/wespeaker/__init__.py):
16 kHz PCM scaled by 32768, 80 mel bands, 25 ms Hamming frames, 10 ms shift,
power-of-two FFT, snipped edges, zero dither, no energy feature and global mean
centering per frequency. This is not a managed audio-frontend qualification.

The native oracle uses NumPy 2.2.4, ORT 1.29.0, Torch and Torchaudio 2.11.0+cpu.
ORT runs sequentially with one intra/inter-op thread, all graph optimizations
and no spinning. The generator checks the model/source hashes and independently
generates all 16 NPY inputs/outputs. Provide a local pyannote-audio checkout at
the exact revision above and the [recorded PCM manifest](../../audio/README.md):

```powershell
python tests/pyannote/embedding/generate_reference.py --models models/pyannote-embedding --speech artifacts/whisper-files-20260919/speech/audio.json --reference-source external/pyannote-audio --output artifacts/embedding-new/reference
dotnet build tests/pyannote/embedding/EmbeddingReplay.csproj -c Release --tl:off --nologo -v minimal
dotnet tests/pyannote/embedding/bin/Release/net10.0/EmbeddingReplay.dll models/pyannote-embedding/embedding_encoder.onnx artifacts/embedding-new/reference/manifest.json artifacts/embedding-new/managed.json
python tests/pyannote/embedding/audit.py --manifest artifacts/embedding-new/reference/manifest.json --result artifacts/embedding-new/managed.json --output artifacts/embedding-new/audit.json
```

Every output path must be new. Missing assets or mismatches fail explicitly.
Ordinary offline unit tests do not download or require the large assets.

The replay executes all eight cases through both facade and private Memory
contexts, preserves input/held-output bytes across execution and Reset, rejects
missing/rank/width errors and verifies recovery. It checks 16 arrays containing
2,764,800 values with finite scaled error
`abs(managed-native)/max(1,abs(native)) <= 1e-4`. Full managed arrays and their
hashes/shapes are saved beside the report; the NumPy auditor independently
recomputes every comparison and repeat check. Maximum observed local error is
`6.854534149169922e-6`. Default and preferred experimental settings produce
identical bits locally. AMD confirmation remains separate.

The packed-weight budget is 64 MiB; it is not a total memory limit. Timing and
process memory observations are diagnostic. Passing the lane does not establish
complete embeddings, diarization accuracy, overlap handling or a service memory
bound, and does not erase the separate segmentation model's retained silence
failures.
