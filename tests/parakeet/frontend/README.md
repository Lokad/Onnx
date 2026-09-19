# Parakeet PCM frontend replay

This opt-in lane runs the pinned `nemo128.onnx` graph entirely in managed code.
It compares every feature and length against native ORT, including three recorded
speech samples, silence, tones, impulses, several noise lengths, and a batch with
different valid lengths. It also checks invalid-input recovery and retained
input/output ownership. It does not decode tokens or produce a transcript.

Stage `nemo128.onnx` from `istupakov/parakeet-tdt-0.6b-v3-onnx` revision
`8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce`. Its SHA256 is
`a9fde1486ebfcc08f328d75ad4610c67835fea58c73ba57e3209a6f6cf019e9f`.
Use the development dependencies in `../requirements.txt`.

The recorded-speech input is a JSON manifest with `sample_rate: 16000` and
`cases` named `english-16k`, `french-44k-stereo`, and `jfk-48k-stereo`.
Each case supplies `pcm` (a relative NPY filename) and `pcm_sha256`. These must
be finite mono float32 arrays at 16 kHz; the case names describe the source
recordings before conversion. The generator retains all source metadata in its
result. The current qualified recordings are retained in
`artifacts/whisper-files-20260919/speech/audio.json`, with their original pinned
sources and conversion provenance. Another speech manifest is a new qualification
set, not evidence that the original recordings were reproduced.

From the repository root:

```powershell
python tests/parakeet/frontend/generate_reference.py --model models/parakeet-tdt-0.6b-v3/nemo128.onnx --speech-manifest artifacts/whisper-files-20260919/speech/audio.json --output artifacts/parakeet-frontend-new/reference
dotnet build tests/parakeet/frontend/FrontendReplay.csproj -c Release --tl:off --nologo -v minimal
dotnet tests/parakeet/frontend/bin/Release/net10.0/FrontendReplay.dll models/parakeet-tdt-0.6b-v3/nemo128.onnx artifacts/parakeet-frontend-new/reference/manifest.json artifacts/parakeet-frontend-new/managed.json
python tests/parakeet/frontend/audit.py --manifest artifacts/parakeet-frontend-new/reference/manifest.json --result artifacts/parakeet-frontend-new/managed.json --output artifacts/parakeet-frontend-new/audit.json
```

Success requires 26 complete output comparisons, one invalid-input rejection,
bit-identical recovery, exact int64 lengths, and finite features with scaled
error at most `1e-4`. Output paths must be new. All managed arrays are written
beside the report as raw little-endian files with hashes, shapes and types.
The report binds the pinned graph, generator, reference files, actual core and
runner assemblies, runtime and settings. Native ORT is refused in the managed
process. Timings and peak memory are observations, not benchmark qualification.

The graph uses float pre-emphasis, reflection padding, double-precision 512-point
STFT, its embedded mel weights, a guarded natural logarithm and masked sample
variance normalization. The tests include 257-sample input and longer recordings.
Shorter input and invalid per-item sample lengths are not qualified by this lane.
Full TDT token/duration decoding, vocabulary handling, and complete transcription
remain separate requirements.

Small operator fixtures can be regenerated independently:

```powershell
python tests/audio/generate_signal_reference.py --output artifacts/signal-reference-new.json
```

Compare the output before deliberately replacing the committed
`tests/Lokad.Onnx.Backend.Tests/fixtures/signal-reference.json`. Complex STFT
window/batch cases use both ONNX's reference evaluator and NumPy because the
tested native runtime has complex-window and pointer-offset defects. Repeated
and singleton reflection use NumPy. Ordinary real STFT and the remaining
fixtures use native ORT. Integer sum-of-squares follows the native widening
and saturating conversion behavior. Float sums use the same fixed four-lane
ordering as the existing InstanceNormalization kernel; silence normalization
is sensitive to the rounding of the preceding sum.
