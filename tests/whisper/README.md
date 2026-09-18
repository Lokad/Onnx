# Whisper Large V3 Turbo component qualification

This opt-in lane runs the FP32 split decoder export from
`onnx-community/whisper-large-v3-turbo` at revision
`360ebcde2559d60bb474678be3c1de9ef347d01a`.
`decoder-assets.json` contains its exact file hashes and lengths and is embedded
in the managed runner. It reuses the validated `tests/Shared/NpySupport.cs`
reader selectively imported from the voice branch.

Stage the pinned files locally, then run from the repository root:

```powershell
hf download onnx-community/whisper-large-v3-turbo onnx/decoder_model.onnx onnx/decoder_with_past_model.onnx generation_config.json --revision 360ebcde2559d60bb474678be3c1de9ef347d01a --local-dir models/whisper-large-v3-turbo
python -m pip install -r tests/whisper/requirements.txt
pwsh -NoProfile -File eng/test-whisper.ps1 -Output artifacts/whisper-decoder-check
```

The output directory must be new; existing evidence is never replaced. No model
download occurs in the test script. The project is outside the ordinary solution
test run because the two decoder assets total about 1.32 GB.

The native Python generator creates two scenarios with synthetic encoder states
of length 7 and 1500, prefixes of length 1 and 4, and four cached steps each.
Every output, including all self/cross-attention keys and values, is checked for
shape, dtype, finite values and scaled absolute error at most `1e-4`:
`abs(actual-reference) / max(1, abs(reference))`.
All tensor files, model files and the fixture manifest have recorded SHA-256s.

The managed process loads no native ORT library. Native ORT is a development
oracle only. Managed cached calls consume earlier **managed** outputs; the runner
rejects substituting native cache files. Each step verifies input/cache immutability
and keeps all earlier outputs alive through later calls and resets. Both scenarios
share the loaded plans to exercise fresh-request state. `managed.json` records
every comparison, all configured `LOKAD_` environment switches and the actual
core/runner hashes. Execute times are diagnostics,
with no benchmark or calibration claim.

This establishes decoder component behavior on these fixtures. It does not qualify
the encoder, audio transcription, greedy generation, maximum cache lengths,
timestamp alignment, quantized exports or merged graphs. Those remain in PLAN.md.
No complete Whisper support is claimed by this lane.

The Data project separately provides `WhisperAudio.LogMelSpectrogram(samples, 16000)`.
Pass mono float PCM, normally in `[-1,1]`; it returns `[1,128,3000]` float features
for the pinned encoder. It pads short clips with silence and truncates clips over
30 seconds. Empty input is silence. Other sample rates and nonfinite samples are
rejected. It does not resample, mix channels, normalize volume, or decode audio
files. Calls own their scratch and outputs and can run concurrently.

The implementation uses the Whisper frontend contract: periodic Hann window,
centered reflection after right-padding/truncation, exact 400-point transform,
160-sample hops, Slaney mel scale with area normalization, log10 power floor
`1e-10`, clipping at the global maximum minus eight, and `(log + 4) / 4` scaling.
It has no native or Python dependency. See the
[OpenAI frontend reference](https://github.com/openai/whisper/blob/main/whisper/audio.py).

`WhisperAudioTests` runs in the normal backend suite. It checks independent frozen
NumPy reference values at boundary and interior frames, exact silence, truncation,
padding, rejection, input/output ownership and concurrent calls. Regenerate the
fixture explicitly with the development-only dependencies `numpy==2.2.4`,
`transformers==5.16.1`, and CPU `torch==2.11.0`:

```powershell
python tests/whisper/generate_frontend_reference.py --output artifacts/whisper-frontend-reference --fixture tests/Lokad.Onnx.Backend.Tests/fixtures/whisper-frontend.json
dotnet test tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj -c Release --tl:off --nologo -v minimal --filter FullyQualifiedName~WhisperAudioTests
```

The generator retains full NumPy and Torch feature tensors, PCM, hashes and
library/source identities in the new output directory. The committed fixture
contains selected frames from seven deterministic cases. Its `1e-5` absolute
gate compares against the NumPy frontend. NumPy/Torch differences are recorded
separately; their float arithmetic is not bitwise identical. Frontend agreement
does not establish encoder or transcription agreement.
