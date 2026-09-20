# Model support and qualification

Lokad.Onnx executes these model families in managed C# on the CPU. The audio
APIs use specific FP32 exports and local model files; they do not download models
or delegate inference to native ONNX Runtime. The commands below require the
Release CLI and the assets described in each model's linked instructions.

| Model | Available application behavior | Qualification and remaining limits |
|---|---|---|
| multilingual-e5-small | Embeddings through `ComputationalGraph` or `lonnx run` | Native output checks cover scalar, SIMD and intrinsic execution. The [fresh public-options / ORT comparison](../tests/e5/public-ort-20260919.md) retains three primary cases above the 5% latency target: observed Default gaps are 9.2–12.0%, or 7.5–8.1% with explicit Memory. Eight tokens are within the target margin. Results remain descriptive; historical timing calibration is unresolved. |
| Parakeet TDT 0.6B V3 | Short `Transcribe` API; `TranscribeRecording` and `lonnx transcribe --model-type parakeet --recording` for up to ten minutes with independent windows and explicit boundaries/limits | Short recorded decisions agree locally and on AMD; all 784 AMD arrays pass, while three Windows duration-logit arrays fail. Twenty clean-English clips have 11/559 word errors. Recording API/CLI cases match the native reference on Windows and AMD, including 600-second repeated speech and silence. AMD speech takes 233.855 API seconds; sequence peak sampled RSS is 12.379 GB. Accuracy is limited: connected/leading-silence variants have 4/160 and 18/160 word errors; DC-offset hard-cut stress has 84/160. Independent long/noisy speech and Windows numerical acceptance remain open. [Short usage](../tests/parakeet/transcribe/README.md), [recording policy](../tests/parakeet/recording/README.md), [AMD evidence](../tests/parakeet/recording-amd/results-20260919.md). |
| Whisper Large V3 Turbo | `WhisperTranscriber.Transcribe` for up to 30 seconds; `TranscribeRecording` and CLI `--recording` for bounded recordings up to ten minutes, with segment timestamps | Short recorded decisions agree locally and on AMD; decoder/cache components pass their tensor checks. The twenty-recording English check records 10/559 word errors. Recording mode matches native tokens, timestamps, seek and stop decisions on constructed 69/72-second fixtures, with work limits and CLI/API agreement. A separate 600-second repeated-speech API request and repeat match native on Windows and AMD: 26 windows, 64 segments and 1,865 tokens. Sequence peaks are 13.851 GB and 11.720 GB sampled RSS, respectively; silence, ownership and recovery checks pass. These are finite resource observations. Independent long/noisy/multilingual accuracy and full encoder/logit numerical agreement remain open. Language is explicit; automatic detection, translation and word timestamps are not implemented. [Short usage](../tests/whisper/README.md), [recording policy](../tests/whisper/recording/README.md), [AMD API/CLI evidence](../tests/whisper/recording-amd/results-20260919.md), [maximum-speech API evidence](../tests/whisper/maximum-speech/results-20260919.md). |
| pyannote Community-1 | `Community1Diarizer` and `lonnx diarize`; segmentation, speaker embeddings, automatic clustering and ordinary/exclusive speaker intervals | Connected recorded application decisions agree locally and on AMD. A labeled 30-second two-speaker example matches native scoring; it is a small accuracy sample. Filterbank and some earlier silence tensors still fail the full numerical gate. The API accepts up to ten minutes; one repeated-clip Windows request has finite resource/application proof, without a complete long intermediate trace or AMD maximum-duration proof. [Pipeline](../tests/pyannote/diarization/README.md), [dialogue and long-request evidence](../tests/pyannote/dialogue/README.md). |

For example, after staging the pinned assets:

```powershell
lonnx.cmd transcribe models/parakeet-tdt-0.6b-v3 recording.wav --model-type parakeet --json
lonnx.cmd transcribe models/parakeet-tdt-0.6b-v3 longer.wav --model-type parakeet --recording --json
lonnx.cmd transcribe models/whisper-large-v3-turbo recording.wav --language en --json
lonnx.cmd transcribe models/whisper-large-v3-turbo longer.wav --language en --recording --json
```

Both transcription commands accept supported mono/stereo RIFF/WAVE files and
convert them to mono 16 kHz PCM. Default transcription rejects inputs longer than
30 seconds; both explicit recording modes accept up to ten minutes and report
observed windows and the processed position. Whisper reports committed timed
segments; Parakeet exposes quiet/hard boundaries and joins completed window text.
Token/window limits produce an explicitly marked partial result. Cancellation takes
effect between graph calls; an in-flight model call finishes first. The public
PCM APIs expect callers to supply the required sample rate directly.

Native agreement of a transcript or speaker timeline does not establish full
tensor agreement or accuracy against human annotations. The numerical gate is
`abs(actual-reference) / max(1, abs(reference)) <= 1e-4`, with exact checks for
integer decisions and shape. Recorded failures remain failures under this rule.
Broader multilingual, noisy, overlapping and long-recording accuracy needs
separate evidence.

The [labeled ASR observation](../tests/audio/accuracy/results-20260919.md)
retains every error on its fixed twenty-recording subset. Its one-word
difference between the recognizers is too small to establish a general ranking.

The subsequent [Whisper numerical localization](../tests/whisper/numerical-20260919.md)
retains the failed full-pipeline gate: 21 encoder arrays and 405 logit arrays
exceed the tolerance. All 707 decoder-logit arrays pass in a separate diagnostic
using native encoder states, with independently advanced managed caches. Both
paths match every token decision. This narrows the remaining investigation to
the source of the differing encoder states; diagnostic success does not qualify
the complete managed pipeline.

DINOv3, ResNet50 and GPT-2 also have complete-output shared-core regression
fixtures, including independently carried GPT states and ownership/recovery
checks. They exercise common kernels beyond e5 and audio; these fixtures are
not general-purpose image or text generation application APIs.

See [runtime options](runtime-options.md) for enabled optimizations, diagnostic
controls and explicit intermediate-lifetime selection. Packed-weight and
released-array cache limits bound those caches individually, not total process
memory. Published resource observations apply to their stated recording,
hardware, revision and request count.
