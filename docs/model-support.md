# Model support and qualification

Lokad.Onnx executes these model families in managed C# on the CPU. The audio
APIs use specific FP32 exports and local model files; they do not download models
or delegate inference to native ONNX Runtime. The commands below require the
Release CLI and the assets described in each model's linked instructions.

Audio orchestration APIs live in the repository's `Lokad.Onnx.Data` project;
the core `Lokad.Onnx` NuGet package does not include that assembly. The
[current core package check](../tests/package-fingerprint-20260920.md) verifies a
fresh package restore and real graph/import execution by an independent app,
with the fingerprint cache both disabled and enabled.

| Model | Available application behavior | Qualification and remaining limits |
|---|---|---|
| multilingual-e5-small | Embeddings through `ComputationalGraph` or `lonnx run` | Native output checks cover scalar, SIMD and intrinsic execution. The [latest independent-deployment / ORT observations](../tests/e5/fingerprint-deployment/aa-results-20260920.md) retain three primary cases above the 5% latency target: observed Default gaps are 7.0–12.4%, or 6.5–10.9% with explicit Memory, averaging the three identical managed roles. Eight tokens are within the target margin. All output/resource checks pass, but timing controls fail; results remain descriptive and the cache stays off by default. |
| Parakeet TDT 0.6B V3 | Short `Transcribe` API; `TranscribeRecording` and `lonnx transcribe --model-type parakeet --recording` for up to ten minutes with independent windows and explicit boundaries/limits | Short recorded decisions agree locally and on AMD; all 784 AMD arrays pass, while three Windows duration-logit arrays fail. Twenty clean-English clips have 11/559 word errors. A separate five-language FLEURS check has 26/418 clean and 36/418 artificial-noise word errors, identical to ORT, with all 41 public results matching. Recording API/CLI cases match the native reference on Windows and AMD, including 600-second repeated speech and silence. AMD speech takes 233.855 API seconds; sequence peak sampled RSS is 12.379 GB. Accuracy is limited: connected/leading-silence variants have 4/160 and 18/160 word errors; DC-offset hard-cut stress has 84/160. Two natural ten-minute AMI meetings have 570/2,461 word errors (23.1613% WER), identical to ORT; all three public comparisons and ownership/recovery checks pass. Broader natural-noise/conversation coverage and Windows numerical acceptance remain open. [Natural meeting evidence](../tests/audio/natural-meetings/results-20260920.md). [Five-language evidence](../tests/audio/multilingual/results-20260920.md). [Short usage](../tests/parakeet/transcribe/README.md), [recording policy](../tests/parakeet/recording/README.md), [AMD evidence](../tests/parakeet/recording-amd/results-20260919.md). |
| Whisper Large V3 Turbo | `WhisperTranscriber.Transcribe` for up to 30 seconds; `TranscribeRecording` and CLI `--recording` for bounded recordings up to ten minutes, with segment timestamps | Short recorded decisions agree locally and on AMD; decoder/cache components pass their tensor checks. The twenty-recording English check records 10/559 word errors. A separate five-language FLEURS check has 17/418 clean and 38/418 artificial-noise word errors, identical to ORT, with all 41 public results matching. Recording mode matches native tokens, timestamps, seek and stop decisions on constructed 69/72-second fixtures, with work limits and CLI/API agreement. A separate 600-second repeated-speech API request and repeat match native on Windows and AMD: 26 windows, 64 segments and 1,865 tokens. Sequence peaks are 13.851 GB and 11.720 GB sampled RSS, respectively; silence, ownership and recovery checks pass. These are finite resource observations. Two natural ten-minute AMI meetings have 668/2,461 word errors (27.1434% WER), identical to ORT; all three public comparisons and ownership/recovery checks pass. Native Whisper uses the declared Linux continuation after a preserved Windows memory failure and retry refusal. Broader natural-noise/conversation/language coverage and full encoder/logit numerical agreement remain open. [Natural meeting evidence](../tests/audio/natural-meetings/results-20260920.md). [Five-language evidence](../tests/audio/multilingual/results-20260920.md). Language is explicit; automatic detection, translation and word timestamps are not implemented. [Short usage](../tests/whisper/README.md), [recording policy](../tests/whisper/recording/README.md), [AMD API/CLI evidence](../tests/whisper/recording-amd/results-20260919.md), [maximum-speech API evidence](../tests/whisper/maximum-speech/results-20260919.md). |
| pyannote Community-1 | `Community1Diarizer` and `lonnx diarize`; segmentation, speaker embeddings, automatic clustering and ordinary/exclusive speaker intervals, through ten minutes | Connected recorded decisions agree locally and on AMD. Two natural ten-minute AMI meetings have aggregate ordinary/exclusive DER of 21.4593%/24.7763%, identical to the ORT application; all three public request comparisons, ownership and recovery pass, with maximum centroid scaled error 8.93205e-7. Managed AMD requests take 1,284.984/1,316.065 API seconds; sequence peak sampled RSS is 3.583 GB. These are two fixed excerpts, not corpus-wide accuracy or matched cross-host latency evidence. Constructed ten-minute requests also have finite application/resource proof on both hosts. Filterbank and some earlier silence tensors still fail the full numerical gate; complete long intermediate traces and broader accuracy remain open. [Natural meeting results](../tests/pyannote/natural-meetings/results-20260920.md), [pipeline](../tests/pyannote/diarization/README.md), [dialogue and Windows long-request evidence](../tests/pyannote/dialogue/README.md), [AMD constructed maximum-duration evidence](../tests/pyannote/maximum-amd/results-20260919.md). |

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
The [five-language ASR check](../tests/audio/multilingual/results-20260920.md)
adds twenty FLEURS read recordings and their deterministic 10 dB noise variants.
Both recognizers match ORT on all public decisions. This small, partly correlated
sample does not establish general natural-noise or conversation accuracy. The
separate [natural meeting evaluation](../tests/audio/natural-meetings/results-20260920.md)
adds two independent ten-minute, four-speaker excerpts. Its chronological
mixed-speaker reference retains fillers and truncated words; overlap makes
reference ordering ambiguous. The results apply to those excerpts and do not
establish an overall recognizer ranking.

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

The [full-corpus encoder input comparison](../tests/whisper/input-cross-isolated/results-20260920.md)
then reproduces all 42 saved baselines and compares both engines on identical
features. All 21 cases still exceed `1e-4` with either feature source, so frontend
rounding alone does not explain the encoder discrepancy. Its 42 finite workers
complete within the unchanged resource guards; this does not qualify a
long-lived transcriber process or identify the more accurate FP32 engine.
The [saved-frame analysis](../tests/whisper/frame-distribution/results-20260920.md)
places about 97% of failing values beyond the original recordings, but also
retains thousands of failures before their ends. No encoder positions are
excluded from qualification.

The subsequent [selected natural-case traces](../tests/whisper/trace-selected/results-20260920.md)
first exceed `1e-4` at encoder layer 20 in all six case/feature comparisons.
Managed final outputs retain their unmodified bits; native instrumentation
differences remain below the threshold, with maximum `6.41e-5`. This nominates
layer 20 for controlled same-input investigation. It does not identify a faulty
operator or replace full-corpus qualification. All 656 intermediate arrays,
including the repeated first case, are retained and independently checked.

The [controlled layer-20 comparison](../tests/whisper/layer20-cross/results-20260920.md)
then reproduces all sixteen same-engine traced boundaries bit for bit. Across
all twelve intermediate outputs, both engines agree on identical incoming
arrays: maximum scaled difference `4.33326e-5`. Changing only the saved incoming
array produces layer-end differences of roughly `6.6e-4` to `1.3e-3` within
either engine. This localizes amplification of existing input differences in
these selected cases. It does not establish which earlier arithmetic is more
accurate or resolve the full encoder/logit numerical failures. All 384 complete
arrays and the first-case repeats remain available.

The [higher-precision K/fc1 diagnostic](../tests/whisper/natural-projection-reference/results-20260920.md)
checks each projection against a float64 calculation on its own saved normalized
input and original weights. All 64 comparisons stay below `1e-4`, with maximum
`5.94843e-6`; all repeats match and 1,727 scalar checks pass. The larger diagonal
differences at these stages are reproduced by propagating the differing inputs.
This narrows the investigation but leaves the full-encoder numerical failures
and the accuracy of earlier arithmetic unresolved.

DINOv3, ResNet50 and GPT-2 also have complete-output shared-core regression
fixtures, including independently carried GPT states and ownership/recovery
checks. They exercise common kernels beyond e5 and audio; these fixtures are
not general-purpose image or text generation application APIs.

See [runtime options](runtime-options.md) for enabled optimizations, diagnostic
controls and explicit intermediate-lifetime selection. Packed-weight and
released-array cache limits bound those caches individually, not total process
memory. Published resource observations apply to their stated recording,
hardware, revision and request count.
