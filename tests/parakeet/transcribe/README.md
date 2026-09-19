# Managed Parakeet transcription

`ParakeetTranscriber` in `Lokad.Onnx.Data` connects the FP32 Parakeet TDT 0.6B V3
frontend, encoder, recurrent decoder/joint model and vocabulary. Inference is
entirely managed and uses local files. It does not download assets or load native
ONNX Runtime.

Stage the six files pinned in [assets.json](assets.json) under one directory:
`nemo128.onnx`, `encoder-model.onnx`, `encoder-model.onnx.data`,
`decoder_joint-model.onnx`, `config.json` and `vocab.txt`. The export is
`istupakov/parakeet-tdt-0.6b-v3-onnx`, revision
`8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce`. Its encoder weights occupy about 2.4 GB.

```powershell
build.cmd
lonnx.cmd transcribe models/parakeet-tdt-0.6b-v3 recording.wav --model-type parakeet
lonnx.cmd transcribe models/parakeet-tdt-0.6b-v3 recording.wav --model-type parakeet --json
```

The CLI accepts mono/stereo PCM or floating-point WAV at 8–192 kHz, mixes to mono
and resamples to 16 kHz. Recordings are limited to 30 seconds. Parakeet recognizes
languages without a language argument; passing `--language` is an error. Whisper
remains the default model type and still requires that argument. Text or JSON goes
to stdout and errors or a token-limit warning go to stderr. Ctrl-C returns 130;
an in-flight graph call completes before cancellation takes effect.

```csharp
var model = new ParakeetTranscriber("models/parakeet-tdt-0.6b-v3");
var result = model.Transcribe(monoPcm, 16000,
    ParakeetTranscriptionOptions.Default, CancellationToken.None);
Console.WriteLine(result.Text);
```

The API accepts finite mono 16 kHz samples, serializes requests on each instance,
and preserves caller input and earlier results. Each request has independent
execution contexts and hidden/cell states. Encoder and decoder packed-weight
caches are bounded to 256 and 64 MiB; these are cache limits, not total-memory
limits. Exact digital silence, including empty input, returns `SilentInput`
without graph execution. Nonsilent input needs at least 257 samples. This silence
bypass is not a voice activity detector.

`TokenIds`, `FrameIndices` and `DurationFrames` are immutable lists with one entry
per emitted token. Repeated token IDs are retained. Encoder frames represent
nominal 80 ms intervals; they are model alignment metadata, not qualified word
timestamps. Blank predictions still contribute to `DecoderCalls`. Positive
duration predictions advance 1–4 frames. A blank with zero duration or the
per-frame token cap forces one frame of progress. Default limits are 4096 tokens
per request and ten per frame. `TokenLimit` distinguishes truncation from
`EndOfAudio`; `--max-tokens` can lower the request limit.

## Recorded agreement and remaining limits

The local replay agrees exactly with native ORT 1.29.0 and the pinned onnx-asr
greedy loop on English, French and JFK recordings, token/frame limits, silence
and a repeated request: seven cases, including text, tokens, frame positions,
durations, call counts and stop reasons. Six invalid/canceled requests are followed
by successful recovery. Six actual CLI checks also agree, including stereo
44.1/48 kHz WAV conversion, JSON, text and truncation diagnostics.

Full numerical qualification remains **failed**. All 784 saved output arrays
(3,090,494 values) are compared, including every recurrent state. Three arrays
exceed the unchanged `abs(actual-reference)/max(1,abs(reference)) <= 1e-4` gate.
They reproduce the same duration-zero logit at English decoder call 26 across
the normal, one-token-per-frame and repeat cases; maximum error is
`2.321004867553711e-4`. All application decisions still agree. Native ORT given
managed encoder output reproduces almost all of the discrepancy, localizing it
to propagated frontend/encoder differences. This evidence does not establish
general speech accuracy, long-audio behavior, INT8 support or AMD qualification
of the complete pipeline. A local process peak near 6.3 GB is a finite observation.

## Reproduce the opt-in full-array replay

Install the pinned packages in `../requirements.txt`. Supply the recorded PCM
manifest produced by the Whisper file-input lane and a checkout of
`https://github.com/istupakov/onnx-asr` at commit
`675f0e68c24d846ee1775743e92d9b4ed452380e`. The generator checks source and asset
hashes and executes the pinned upstream loop with independently carried native
states. The managed replay advances its own states; saved native states never
feed its recurrence.

```powershell
python tests/parakeet/transcribe/generate_reference.py --models models/parakeet-tdt-0.6b-v3 --speech artifacts/whisper-files-20260919/speech/audio.json --reference-source external/onnx-asr --output artifacts/parakeet-transcribe-new/reference
dotnet build tests/parakeet/transcribe/TranscribeReplay.csproj -c Release --tl:off --nologo -v minimal
dotnet tests/parakeet/transcribe/bin/Release/net10.0/TranscribeReplay.dll models/parakeet-tdt-0.6b-v3 artifacts/parakeet-transcribe-new/reference/manifest.json artifacts/parakeet-transcribe-new/managed.json
python tests/parakeet/transcribe/audit.py --manifest artifacts/parakeet-transcribe-new/reference/manifest.json --result artifacts/parakeet-transcribe-new/managed.json --output artifacts/parakeet-transcribe-new/audit.json
```

Use fresh output paths. Both replay and independent NumPy audit return 1 when
the numerical gate fails, even with `application_passed: true`.
`audit_consistent: true` means the saved evidence matches the report; inspect
`numeric_gate_passed` separately. Full raw managed arrays are retained beside
the report in `<result.json>.tensors`, with shapes, dtypes and hashes. The replay
also verifies pinned fixtures, every native token/state transition, held tensor
ownership, repeated-result ownership and absence of native ORT in its process.
