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

## Managed transcription API

For longer recordings, use the separate [recording API and CLI mode](recording/README.md).
It adds segment timestamps and boundary recovery with explicit input/work limits.
Its Windows application evidence is separate from the short-model and numerical
qualifications described below.

`WhisperTranscriber` in `Lokad.Onnx.Data` composes the frontend and three local
FP32 split graphs. It accepts finite mono 16 kHz PCM up to 30 seconds:

```csharp
var model = new WhisperTranscriber("models/whisper-large-v3-turbo");
var result = model.Transcribe(samples, 16000,
    WhisperTranscriptionOptions.ForLanguage("fr"), CancellationToken.None);
Console.WriteLine(result.Text);
```

The same instance serializes requests and reuses three execution contexts. Each
request creates independent attention-cache state, and returned results remain
owned by the caller. Released-array caches retain at most 512 MiB for the encoder
and 128 MiB for each decoder; these are payload budgets, not a process-memory
ceiling. Identical private decoder weights share backing storage after exact byte
comparison. The [AMD endurance qualification](weight-sharing-v2/results-20260920.md)
and [recording/concurrency/recovery checks](memory-contracts-amd/results-20260921.md)
cover this behavior. The API never downloads assets or loads native ORT.
Language is explicit; the initial policy is greedy transcription without timestamps.
The generation configuration supplies the prefix and suppression lists. All non-text
control/timestamp tokens are excluded from generated text. The result preserves
actual token IDs and distinguishes EOS, token-limit termination and exact digital
silence. Requests longer than 30 seconds are rejected, not silently truncated.
The WAV/CLI path below performs decoding and resampling before this API. A
long-audio segmentation policy remains separate work.

The three loaded graphs limit their optional packed weight copies to 256 MiB for
the encoder and 64 MiB for each decoder. Original weights, folded transposes and
request buffers are outside these limits. Other graph callers retain their existing
packing defaults; `OnnxImport.Load(path, maximumPackedWeightBytes)` and
`Model.Load(description, maximumPackedWeightBytes)` allow an explicit per-graph
budget, with zero disabling packing. This is not a process memory ceiling.
Transcription uses `ExecutionOptions.Memory`, which releases eligible dead
intermediate reshape bindings so their storage can be reused within the call.
It preserves inputs and returned outputs. It does not force garbage collection
or change process-wide runtime settings.

Empty or exactly zero PCM returns empty text without inference; its model
probabilities are null. The pinned native model otherwise emits “you” on the
one-second all-zero fixture, and its no-speech probability does not catch it.
No energy threshold is used. For nonzero audio, the default policy skips text when
the raw first-position no-speech probability exceeds 0.6 unless average generated
log probability exceeds -1.0. These thresholds follow the
[OpenAI transcription policy](https://github.com/openai/whisper/blob/main/whisper/transcribe.py).
Either threshold can be disabled with null. This policy does not guarantee that
background noise or every nonspeech clip produces empty text.

The ordinary `WhisperGenerationTests` cover frozen independent token decoding,
UTF-8 bytes split across tokens, English/French prefixes, cache progression and
request isolation, suppression, EOS/limits, probabilities, cancellation and failures.
Regenerate its compact fixture with `tokenizers==0.23.2`:

```powershell
python tests/whisper/generate_text_reference.py --output tests/Lokad.Onnx.Backend.Tests/fixtures/whisper-text.json
```

An opt-in complete API replay uses `TranscribeReplay.csproj`. Its embedded
`transcription-assets.json` pins every model/metadata file. The native oracle takes
an audio manifest with hash-bound mono PCM and independently generated NumPy
features. Each case includes `name`, `language`, `max_new_tokens`, `samples`, `pcm`,
`pcm_sha256`, `features` and `features_sha256`; file paths are relative to the
manifest. The manifest declares `sample_rate: 16000`. The frontend recipe and
original audio/resampling identities belong in the manifest as provenance.

```powershell
python tests/whisper/generate_transcription_reference.py --models models/whisper-large-v3-turbo --audio <audio.json> --output <new-reference-directory>
dotnet build tests/whisper/TranscribeReplay.csproj -c Release --tl:off --nologo -v minimal
dotnet tests/whisper/bin/Release/net10.0/TranscribeReplay.dll models/whisper-large-v3-turbo <new-reference-directory>/manifest.json <new-result.json>
```

The managed replay accepts PCM and computes its own features, encoder output,
caches and token choices. It compares complete token/text/stop decisions, verifies
input bytes, repeats the first request after other clips and invalid/canceled calls,
and rejects native ORT in its process. It records confidence and peak memory.
This is application agreement, not a replacement for the existing `1e-4` tensor
checks. Full encoder/logit numerical qualification and broader audio acceptance
remain unresolved; successful transcript fixtures must not be reported as proving
those separate requirements.


The decoder boundary lane explicitly tests a 447-token prefix followed by a cached
step to 448, a separate 448-token prefix, and both first/cached rejection beyond
that position table. It checks complete logits/caches and preservation of earlier
outputs and inputs after failed requests:

```powershell
python tests/whisper/generate_decoder_reference.py --models models/whisper-large-v3-turbo --output <new-boundary-directory> --boundary
dotnet build tests/whisper/DecoderReplay.csproj -c Release --tl:off --nologo -v minimal
dotnet tests/whisper/bin/Release/net10.0/DecoderReplay.dll models/whisper-large-v3-turbo <new-boundary-directory>/manifest.json <new-boundary-result.json>
```

The ordinary generator still produces its original two scenarios and 106 output
comparisons. Boundary fixtures add 43 complete output comparisons and two expected
position-limit failures; the numerical tolerance remains `1e-4`.

## Transcribing WAV files

Build the Release CLI and stage the pinned local model directory, including the
encoder sidecar and tokenizer/generation metadata. Then run:

```powershell
lonnx.cmd transcribe models/whisper-large-v3-turbo recording.wav --language en
lonnx.cmd transcribe models/whisper-large-v3-turbo recording.wav --language fr --max-tokens 444 --json
```

The command accepts mono/stereo, little-endian RIFF/WAVE recordings up to 30
seconds. PCM8/16/24/32 and IEEE float32/64 (including extensible subtypes) are
supported at 8000..192000 Hz. Stereo channels are averaged and a managed,
anti-aliasing FIR resampler converts to 16 kHz. It does not clip or normalize
volume. Invalid/nonfinite, compressed, RF64 and big-endian files are rejected.
No external audio executable or model download is invoked.

Plain stdout contains the transcript. `--json` emits `text`, `token_ids`,
`stop_reason`, `skipped_as_no_speech`, `no_speech_probability` and
`average_log_probability`; diagnostics use stderr. Language is required and
must occur in the model's generation configuration. There are no timestamps,
translation or automatic language detection. Token-limit termination is a
successful but potentially incomplete result, distinguished from EOS and silence.
Long recordings are rejected, not truncated. Cancellation returns exit 130
after the current conversion/model call finishes.

The reader follows the [Microsoft RIFF container format](https://learn.microsoft.com/en-us/windows/win32/xaudio2/resource-interchange-file-format--riff-)
and [extensible sample representation](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ksmedia/ns-ksmedia-waveformatextensible).
The resampler's explicit centered FIR contract is checked against independent
[SciPy polyphase conversion](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)
and analytic passband/alias-rejection tests. Regenerate its small full-array
fixture with the development-only pinned dependencies:

```powershell
python -m pip install -r tests/whisper/audio-requirements.txt
python tests/whisper/generate_resampling_reference.py --output <new-reference.json>
```

Compare and deliberately replace `tests/Lokad.Onnx.Backend.Tests/fixtures/audio-resampling.json`
when regenerating. This input-path validation and transcript agreement do not
replace the unresolved full encoder/logit `1e-4` gate or broad held-out speech
accuracy, long-audio, and sustained-memory qualification.

A separate [twenty-recording labeled check](../audio/accuracy/results-20260919.md)
matches all native application decisions and measures 10 word errors in 559
reference words (1.7889%) on clean English speech. This small subset does not
resolve the full numerical gate or establish broader recognition accuracy.

The [real-speech numerical localization](numerical-20260919.md) checks complete
features, encoder states and logits on those twenty recordings plus a repeat.
All 707 decoder-logit arrays pass when supplied the native encoder states; the
fully managed path retains 21 failed encoder arrays and 405 failed logit arrays.
Both paths independently advance their own caches and match all native token
and application decisions. This identifies the differing encoder states as the
source of the observed decoder disagreement, without distinguishing frontend
perturbations from encoder arithmetic or replacing the failed full-model gate.

The [shared recorded-audio recipe](../audio/README.md) recreates all five WAV,
PCM and mel-feature fixtures with pinned source and output hashes. Its
`audio.json` can be passed to `generate_transcription_reference.py --audio`.
