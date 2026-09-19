# Managed WeSpeaker filterbanks

`WeSpeakerAudio` in `Lokad.Onnx.Data` prepares the input of the pyannote Community-1 embedding backbone:

```csharp
DenseTensor<float> features = WeSpeakerAudio.LogMelFilterbank(
    monoSamples, 16000, cancellationToken);
// [1, 1 + (monoSamples.Length - 400) / 160, 80]
```

Supply finite normalized mono PCM in `[-1,1]`, from 400 samples through 30 seconds. The helper rejects unsupported rates and lengths, validates the entire input including unused trailing samples, and checks cancellation between frames. Each call owns its scratch and result. It does not resample, pad, truncate, mix channels, remove silence, or perform diarization.

The algorithm follows the pinned [pyannote WeSpeaker frontend](https://github.com/pyannote/pyannote-audio/blob/b749285c5cdd4636b2edc7f766f1352c8dde9369/src/pyannote/audio/models/embedding/wespeaker/__init__.py): scale PCM by 32768, remove frame DC, apply `.97` preemphasis and a nonperiodic Hamming window, compute a 512-point spectrum, accumulate 80 Kaldi mel bands, take natural logs with the float32-epsilon energy floor, then center each feature over the complete request. The managed Fourier computation uses double precision internally.

## Qualification status

The ordinary tests cover complete independent direct-Fourier fixtures, frame boundaries, ownership, concurrent calls, cancellation and invalid requests. These tests require no model assets, Python or native runtime.

The pinned Torch/Torchaudio 2.11.0 CPU reference lane contains 21 complete arrays and 711,680 values. **Its `1e-4` scaled-error gate currently fails on three French-recording values**, at maximum `1.25818e-4`. All other cases pass, including English/JFK recordings, noise, quiet signals, tones, impulses, silence and constant inputs. Default and experimental settings give identical managed outputs locally. The lane deliberately exits 1 and records these failures.

Identical-input diagnostics locate the dominant difference in low-energy Fourier bins: Torch and NumPy double-precision transforms agree closely, while Torch's float32 transform rounds differently. This characterizes the discrepancy without replacing the native reference or increasing the gate. An additional artifact replay through the actual backbone, weighted statistics and an ONNX affine projection passes all downstream arrays and final vectors; it retains the failed frontend gate. Those observations do not establish a production speaker-embedding API, speaker validity, clustering or a diarization timeline.

## Reproduce the full native lane

Use the versions in `pins.json`, the pinned pyannote source checkout, and the shared [recorded audio recipe](../../audio/README.md). The generator verifies source/version identities and creates a new directory; it downloads nothing.

```powershell
python tests/pyannote/frontend/generate_reference.py --reference-source external/pyannote-audio --speech artifacts/audio-new/audio.json --output artifacts/wespeaker-reference-new
dotnet build tests/pyannote/frontend/FrontendReplay.csproj -c Release --tl:off --nologo -v minimal
dotnet tests/pyannote/frontend/bin/Release/net10.0/FrontendReplay.dll artifacts/wespeaker-reference-new artifacts/wespeaker-result-new.json
python tests/pyannote/frontend/audit.py --reference artifacts/wespeaker-reference-new --result artifacts/wespeaker-result-new.json --output artifacts/wespeaker-audit-new.json
```

The managed runner checks embedded reference identities, complete case coverage, file digests, shapes, values, input immutability, repeatability and absence of native ORT. It saves every managed output. The independent NumPy auditor recomputes every comparison and returns 1 for the preserved numerical failure. Metadata or missing-coverage failures must not be interpreted as successful numerical qualification.

`generate_unit_reference.py` creates the small committed mathematical fixture with an independent direct Fourier sum in float64. It refuses to overwrite the fixture and is separate from the native-conformance generator. The mathematical tests do not supersede the native gate.
