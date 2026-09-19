# Model support and qualification

Lokad.Onnx executes these model families in managed C# on the CPU. The audio
APIs use specific FP32 exports and local model files; they do not download models
or delegate inference to native ONNX Runtime. The commands below require the
Release CLI and the assets described in each model's linked instructions.

| Model | Available application behavior | Qualification and remaining limits |
|---|---|---|
| multilingual-e5-small | Embeddings through `ComputationalGraph` or `lonnx run` | Native output checks cover scalar, SIMD and intrinsic execution. The [fresh public-options / ORT comparison](../tests/e5/public-ort-20260919.md) retains three primary cases above the 5% latency target: observed Default gaps are 9.2–12.0%, or 7.5–8.1% with explicit Memory. Eight tokens are within the target margin. Results remain descriptive; historical timing calibration is unresolved. |
| Parakeet TDT 0.6B V3 | `ParakeetTranscriber` and `lonnx transcribe --model-type parakeet`; multilingual greedy transcription of recordings up to 30 seconds | Recorded transcripts, tokens, durations, carried states and recovery agree with native decisions locally and on AMD. The complete AMD replay passes all 784 arrays; Windows retains three duration-logit discrepancies above the numerical gate. A separate twenty-recording clean-English check matches native and records 11/559 word errors (1.9678%). Longer recordings, broader accuracy and the Windows numerical failure remain open. [Usage and evidence](../tests/parakeet/transcribe/README.md). |
| Whisper Large V3 Turbo | `WhisperTranscriber` and `lonnx transcribe --language en`; greedy transcription of recordings up to 30 seconds | Recorded token/text/stop decisions agree locally and on AMD; independent decoder/cache trajectories and position boundaries pass their tensor checks. A separate twenty-recording clean-English check matches native and records 10/559 word errors (1.7889%). Full encoder/logit numerical agreement remains unresolved. Language must be explicit; automatic language detection, translation, timestamps and long-recording orchestration are not implemented. [Usage and evidence](../tests/whisper/README.md). |
| pyannote Community-1 | `Community1Diarizer` and `lonnx diarize`; segmentation, speaker embeddings, automatic clustering and ordinary/exclusive speaker intervals | Connected recorded application decisions agree locally and on AMD. A labeled 30-second two-speaker example matches native scoring; it is a small accuracy sample. Filterbank and some earlier silence tensors still fail the full numerical gate. The API accepts up to ten minutes; one repeated-clip Windows request has finite resource/application proof, without a complete long intermediate trace or AMD maximum-duration proof. [Pipeline](../tests/pyannote/diarization/README.md), [dialogue and long-request evidence](../tests/pyannote/dialogue/README.md). |

For example, after staging the pinned assets:

```powershell
lonnx.cmd transcribe models/parakeet-tdt-0.6b-v3 recording.wav --model-type parakeet --json
lonnx.cmd transcribe models/whisper-large-v3-turbo recording.wav --language en --json
```

Both transcription commands accept supported mono/stereo RIFF/WAVE files and
convert them to mono 16 kHz PCM. Inputs longer than 30 seconds are rejected.
Token limits produce an explicitly marked partial result. Cancellation takes
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
