# CPU benchmarks

## Current results — 2026-09-19 UTC

The tables here summarize retained measurements for e5, Parakeet, Whisper and
pyannote. Each names its workload, hardware and timing boundary. Earlier tables
below remain historical evidence; do not compare absolute times across hosts,
revisions or protocols. [Model support](docs/model-support.md) describes the
available APIs and their remaining qualification limits.

For Parakeet, pyannote and Whisper Large V3 Turbo, see the
[matched Microsoft ONNX Runtime baselines](#audio-matched-microsoft-onnx-runtime-baselines)
below.

### e5: public execution versus native ORT

AMD EPYC 9V74, CPU 2, .NET 10.0.8, SDK 10.0.204, product `c6bf781`.
Managed execution uses the nine qualified production defaults; Memory is the
explicit intermediate-lifetime option. Native ORT 1.23.2 uses one inference
thread and the same logical CPU. Values are mean public Execute/Run times,
excluding loading and tokenization.

| Tokens | Lokad Default ms | Lokad Memory ms | ORT ms | Default / ORT | Memory / ORT |
|---|---:|---:|---:|---:|---:|
| 8 | 6.1477 | 5.8677 | 6.0201 | 1.0212 | 0.9747 |
| 30 | 16.9856 | 16.6011 | 15.3596 | 1.1059 | 1.0808 |
| 30 padded to 128 | 66.3201 | 65.2998 | 60.7387 | 1.0919 | 1.0751 |
| 128 | 67.8665 | 65.1282 | 60.6045 | 1.1198 | 1.0746 |
| 512 | 347.2517 | 337.0893 | 284.5694 | 1.2203 | 1.1846 |

These are descriptive results from ninety fresh processes and 5,940 measured
calls, with every sample retained. All output checks pass, maximum scaled error
`1.63913e-6`. Historical fine timing calibration remains unresolved, so these
figures do not establish calibrated parity or confidence bounds. The
[full comparison](tests/e5/public-ort-20260919.md) includes process ranges,
complete-request boundaries, allocation, memory, conditioning and identities.

A subsequent [opt-in zero-block comparison](tests/e5/softmax-zero-product/results-20260919.md)
retains another 5,940 complete-model calls. Padded-128 Execute averages
65.3973 ms for the controls and 64.7893 ms for the candidate: 0.93% lower,
below its prospective 1% requirement. Identical controls also fail their
stability limits in other cases. Correctness passes, but the performance
conclusion is inconclusive and the switch remains off by default. That
experiment supplies no fresh ORT timing and does not replace the table above.

### Audio: matched Microsoft ONNX Runtime baselines

Fresh complete-application measurements on **Windows i7-14700KF, logical CPU 2**,
with .NET 10.0.12 and Microsoft ONNX Runtime **1.29.0**. Product source is
`8732831` (core execution unchanged from `c6bf781`). Both engines run the same
FP32 models and PCM inputs on one logical CPU. ORT uses one intra-op/inter-op
thread, sequential execution and all graph optimizations.

Times include features, neural inference, decoding or automatic clustering,
and owned output construction. Loading, file access and external validation are
excluded. The pyannote baseline combines ORT segmentation/embedding/projection
graphs with pinned Torch/NumPy/SciPy frontend, pooling and clustering code; it
measures an ORT-backed application. Whisper combines a pinned Transformers
NumPy frontend with ORT encoder and decoder graphs. These are complete
application comparisons, not isolated ORT kernels.

| Application / workload | Lokad seconds | ORT seconds | Lokad / ORT | Lokad RTF | ORT RTF |
|---|---:|---:|---:|---:|---:|
| Parakeet, all 20 clips (213.265 s audio) | 60.683 | 29.673 | 2.045 | 0.285 | 0.139 |
| Whisper Large V3 Turbo, same 20 clips | 434.193 | 349.993 | 1.241 | 2.036 | 1.641 |
| pyannote, full 30 s dialogue | 33.629 | 6.408 | 5.248 | 1.121 | 0.214 |
| pyannote, 0–10 s crop | 1.641 | 0.305 | 5.388 | 0.164 | 0.030 |
| pyannote, 10–20 s crop | 1.717 | 0.305 | 5.628 | 0.172 | 0.031 |
| pyannote, 20–30 s crop | 1.609 | 0.306 | 5.255 | 0.161 | 0.031 |

Lower is faster; **Lokad / ORT > 1 means Lokad.Onnx takes longer**. ASR times
are the mean total for the complete twenty-clip corpus, not one average clip.
Pyannote times are per request; its three crops overlap the full dialogue.
RTF is processing time divided by audio duration.

Each engine/model has two fresh processes, each with one complete corpus warmup
and three measured passes: 120 measured calls for each ASR model and 24 pyannote
calls per engine. All 528 measured calls are retained. Every measured and warmup request
passes the application checks, including held-output and input preservation.
These are descriptive results on an active Windows workstation, without a
calibrated confidence or parity claim. The
[Parakeet/pyannote report](tests/audio/comparison/results-20260919.md) and
[Whisper report](tests/audio/whisper-comparison/results-20260919.md) include
process variation, memory, complete evidence identities and known numerical
limitations. Whisper recomputes its frontend from PCM inside every timed call;
both engines pad each clip to the model's thirty-second encoder input.

### Audio: earlier public API observations

The ASR rows use the same twenty clean-English recordings: ten speakers,
213.265 seconds of audio, individual durations 4.07–17.96 seconds. Each model
runs in one fresh process on the Windows i7-14700KF, confined to CPU 2, with
.NET 10.0.12 and product `c6bf781`. Timings cover the public PCM transcription
call, including managed features and decoding. Model construction, file
reading and external result checks are outside the stopwatch.

**Real-time factor (RTF) = processing seconds / audio seconds; lower is faster.**
An RTF below one means processing took less time than the supplied audio's
duration. ASR totals include all twenty first-pass calls, including the first
request's startup effects. The separately retained repeat is excluded from
those totals. These are observations from an accuracy replay, without dedicated
warmup or repeated performance trials.

| Model / application | Workload | API seconds | RTF | Observed process peak GB |
|---|---|---:|---:|---:|
| Parakeet TDT 0.6B V3, transcription | 20 clips / 213.265 audio seconds | 61.260 | 0.287 | 9.594 |
| Whisper Large V3 Turbo, English transcription | Same 20 clips / 213.265 audio seconds | 427.549 | 2.005 | 12.269 |
| pyannote Community-1, complete diarization | One synthetic 600-second recording / 591 windows | 962.051 | 1.603 | 3.421 |

GB are decimal. ASR peaks are sampled process-group RSS across loading, all
twenty requests and the repeat. The pyannote peak is process peak working set
for its resource/recovery sequence. They are finite observations, not memory
ceilings. All models are FP32 and execute through Lokad.Onnx without native ORT.
The Whisper frontend pads each short clip to its thirty-second encoder input;
the ASR models therefore perform different amounts of work on these recordings.

Parakeet and Whisper model construction took 0.981 and 3.897 seconds,
respectively, after external asset verification. Their first recording took
3.097 and 21.475 seconds; repeating it after the other nineteen took 2.217 and
20.611 seconds. These pairs do not constitute warmed latency distributions.

The pyannote row is a separate Windows observation at `21f3e74`, before the
current default selection. It concatenates twenty copies of the same
thirty-second dialogue. Its supervisor did **not** enforce or record CPU
affinity; do not treat it as a matched single-CPU benchmark or compare its RTF
directly with the ASR rows. The complete call includes segmentation, embeddings,
automatic clustering and interval reconstruction. See the
[long-request evidence](tests/pyannote/dialogue/README.md#finite-ten-minute-request).

[Every retained ASR request time and source identity](tests/audio/accuracy/timing-20260919.json)
is available, including the final repeat. Regenerate this summary from existing
local artifacts with
`python tests/audio/accuracy/summarize_timings.py --output <new-summary.json>`;
the tool verifies the original receipts and does not run inference. The native
audio reference generators also perform validation and evidence export, so their
job durations are not comparable inference times. The matched audio measurements
above use dedicated runners and fresh samples; the older observations here are
not used to calculate those ratios.

A separate [AMD Parakeet recording replay](tests/parakeet/recording-amd/results-20260919.md)
qualifies the API and CLI against the retained native application reference.
Its constructed 600-second repeated-speech request completes 22 windows in
233.855 API seconds; the full API sequence peaks at 12.379 GB sampled RSS.
All ten sequential requests, two concurrent silence calls, two CLI calls and
sixteen refusal/recovery checks pass. These are finite correctness/resource
observations, without a fresh AMD ORT timing comparison or independent natural
ten-minute accuracy claim.

The separate [AMD Whisper recording replay](tests/whisper/recording-amd/results-20260919.md)
also passes its API/CLI, limits, ownership and recovery checks against retained
native application decisions. Its 69.455- and 71.825-second constructed speech
requests take 99.768 and 100.789 API seconds; the sequence peaks at 12.230 GB
sampled RSS. All nine recording calls, ten refusal/recovery checks and the
short-API regression pass. This covers 600-second silence, with maximum-duration
speech resources still open. These finite observations supply no fresh AMD ORT
latency ratio.

### Audio accuracy and numerical agreement

| Model | Human-labeled observation | Result | Managed/native application agreement |
|---|---|---|---|
| Parakeet TDT 0.6B V3 | 20 clean-English clips, 559 reference words | 11 word errors; WER 1.9678%; CER 0.4309% | 21/21 requests, including repeat |
| Whisper Large V3 Turbo | Same 20 clips and labels | 10 word errors; WER 1.7889%; CER 0.5966% | 21/21 requests, including repeat |
| pyannote Community-1 | One annotated 30-second, two-speaker dialogue | Ordinary DER 5.2074%; exclusive DER 10.2909% | Exact timelines and scores on Windows and AMD |

WER/CER are word/character edit error rates; DER measures missed, false-alarm
and confused speaker time. The ASR subset and normalization were fixed before
inference. It is a small LibriSpeech diagnostic, not the full test benchmark;
the one-word difference does not establish a general recognizer ranking. The
pyannote score includes overlap, with zero collar and optimal speaker-label
mapping. Its three correlated ten-second excerpts have much worse ordinary DER
(29.23–45.15%), retained in the report. Sources:
[ASR results and every error](tests/audio/accuracy/results-20260919.md),
[diarization metrics and scope](tests/pyannote/dialogue/README.md#recorded-qualification).

Parakeet's new [recording mode](tests/parakeet/recording/results-20260919.md)
matches ORT application decisions on constructed long inputs, limits and repeats.
The connected 69.455-second example has 4/160 word errors (2.50%); adding 2.37
seconds of leading silence raises this to 18/160 (11.25%) in both engines.
A DC-offset stress case forcing hard cuts has 84/160 errors (52.50%); this
also changes the waveform, so it does not isolate boundary effects. A separate
600-second repeated-speech request completes 22 windows in 173.935 API seconds,
with 13.27 GB sampled process-group peak across the full sequence. These are
finite Windows observations at `f568132`, not a matched recording-latency
comparison or independent natural long-speech qualification. The matched ORT
performance ratios above remain based on their dedicated short-audio campaign.

Whisper's newer timestamp-guided recording mode separately matches native
tokens, segments and seek decisions on constructed 69.455- and 71.825-second
examples, three windows each. Both retain five errors in 160 words (3.125% WER);
they share the same underlying speech and are not independent conversation
tests. [Recording qualification](tests/whisper/recording/results-20260919.md)
covers API/CLI agreement, work limits and finite resources, not a warmed latency
comparison or maximum-duration speech qualification.

Application agreement does not erase numerical discrepancies. Parakeet's
[complete AMD replay](tests/parakeet/transcribe/README.md#complete-amd-pipeline-qualification)
passes all 784 arrays at the unchanged `1e-4` scaled-error gate; three Windows
duration-logit arrays still fail. Whisper's
[full-pipeline numerical check](tests/whisper/numerical-20260919.md) retains
21 encoder and 405 logit-array failures despite identical token choices.
The labeled pyannote trace retains 19 failed filterbank values on Windows and
24 on AMD. Broader multilingual, noisy and long-conversation accuracy and
maximum-duration resource qualification remain open.

## Historical results and methodology — through 2026-09-13

**Review status — 2026-09-09:** the results below are historical and must not
be used as a single-core performance baseline. The `defaults` rows compare
Lokad's one-thread default with ORT's default thread pool; their ratios use
unequal CPU resources. The `one-thread` rows disable Lokad SIMD while ORT
remains optimized. The Auto `mode-threads=1` rows match inference thread
counts, but no row enforces CPU affinity.

The `--rows canonical` selection (the default) prints only the matched
single-CPU row; `--rows all` retains the three-condition setup for diagnostics.

The replacement comparison must give **both engines the same one logical CPU**:
verified process affinity, one inference thread, Lokad Auto with available
SIMD/intrinsics, ORT CPU with intra-op 1/inter-op 1 and sequential execution,
and graph optimizations enabled. Repeat measurements on an identified, quiet
core with identical inputs, requested outputs and output-lifetime boundaries.
Scalar diagnostics and multi-core scaling belong in separate, opt-in results.
Until the harness and measurements meet that contract, retain these tables
only as dated diagnostic evidence; they cannot gate performance commits.

The model harness (`tests/Lokad.Onnx.Bench`) compares the managed engine
against native ONNX Runtime (`Microsoft.ML.OnnxRuntime` 1.23.2) sessions on
CPU only, without registering a GPU provider. The package is the CPU build;
GPU execution requires a different package/provider configuration.
[ORT C# packages](https://onnxruntime.ai/docs/get-started/with-csharp.html)

## Canonical single-CPU baseline — 2026-09-13

Measured under the contract above on LOKAD-0399 (i7-14700KF, 28 logical
CPUs): verified affinity to logical CPU 4 (efficiency-class 1,
core mask 0x30, core-group 0), one inference thread on each side, Lokad Auto with
SIMD/intrinsics, ORT CPU intra-op 1/inter-op 1 sequential with
ORT_ENABLE_ALL, High performance power scheme left unchanged. Three
independent fresh processes, 3 warmups and 33 timed iterations per engine
per case. Every case validated before and after timed reuse at the
unchanged 1e-4 gate with inputs fingerprinted intact. Machine-readable
artifacts with embedded raw samples live in
`tests/Lokad.Onnx.Bench/baseline/summary-20260913.json`, regenerated from
the per-rep logs by `python eng/parse_baseline.py <rep logs>`.
Confinement ratios were 0.83, 0.90 and 0.98 (single-threaded 2 s busy
loop; the harness fails above 1.3 and warns below 0.8): no warnings, but the
same sub-1.00 band from box background load as on 09-12, so the discard rule below
still applies per session.

Measured at `e0e70b1` with a clean tracked tree (untracked PLAN.md and
.agent scratch only); no product-code differences from the 0.2.0 release
`4495fc6` (`src/` identical). SDK
`10.0.300-preview.0.26177.108`, runtime `.NET 10.0.12`, ORT C# `1.23.2.0`,
Lokad assembly `0.2.0.0`. Asset bytes and hashes per case print in each
rep header and match `ModelManifest.json`; inputs and outputs print there
too (e5 token counts, 224x224 pixels, 4-token GPT-2 prefill).

This refresh replaces the 2026-09-12 CPU-4 table after the B01 harness
campaign landed (commit `e0e70b1`): the reused-context lane now alternates
fairly with public-Execute/ORT instead of running in a separate loop, and
each case additionally records Stopwatch-tick raw samples, a `casedef` line,
a `warmup` record, and a Reset+Execute+Reset repeated-request boundary
(`req`/`reqCtx`). The old table and its summary survive in git history
(`summary-20260912.json` stays in `tests/Lokad.Onnx.Bench/baseline/`).

Cells are Lokad warmed public-Execute median versus ORT warmed-Run median
per rep in milliseconds; the ratio spans the three within-rep median
ratios. Absolute medians still drift with machine settling, but both engines
drift together on most cases, so cross-rep ratio spreads hold to 0.1x except
on the two noisy spots below: compare revisions within shared reps and
alternate their order, never absolute medians across days. Per-rep best, p95,
max, GC, req/reqCtx, tick raws, and load/prepare/first-run/warmup figures
live in the summary JSON.

| Case | rep1 L/ORT ms | rep2 L/ORT ms | rep3 L/ORT ms | Lokad / ORT |
|---|---:|---:|---:|---|
| e5-8tok | 25.0 / 7.9 | 24.1 / 8.1 | 25.3 / 8.1 | 3.0-3.2x |
| e5-30tok | 21.4 / 14.4 | 20.1 / 14.3 | 20.3 / 13.9 | 1.4-1.5x |
| dinov3-224 | 104.3 / 75.5 | 112.4 / 79.3 | 106.9 / 75.3 | 1.4-1.4x |
| resnet50-224 | 125.8 / 52.3 | 132.3 / 55.4 | 195.5 / 54.1 | 2.4-3.6x |
| gpt2-4tok | 45.5 / 35.5 | 43.3 / 35.0 | 42.5 / 33.6 | 1.2-1.3x |

Noise verdict for the freeze: e5-8tok medians reproduce (24–25 ms) but both
engines show 2–7x tails (Lokad best 10.5–11.2, max 44.7–72.5), so it is a
noisy case that can neither establish the 0.2.1 win nor a regression — its
gate must use wide bands. Rep3's resnet50 row caught a mid-rep interference
burst (Lokad median 195.5 vs 125.8/132.3 with p95 384, both engines tailed);
the rep is kept, not discarded, and resnet's frozen band stays 2.4–3.6x until
the variation study explains the burst. E5-30tok, DINOv3 and GPT-2 medians
move within 8% rep-to-rep with ratio spreads of at most 0.1x: freezable as
B01 baselines. GPT-2 reads 1.2–1.3x against 1.4–1.5x on 09-12 because absolute
ORT medians drifted up across days (25–30 ms then, 34–36 ms now); that drift
is not claimed as a Lokad improvement.

Reproduce from the repo root after building Release:

```powershell
dotnet tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll e5 dinov3 resnet50 gpt2 --mode auto --threads 1 --rows canonical --cpu 4 --iters 33
```

Run it three times in fresh processes and regenerate the table with the
parser above; a rep with a confinement warning, a failed case, or a
changed asset hash is discarded and rerun, never averaged in.
## Extended single-CPU regimes — 2026-09-13 (B02)

Same machine, affinity, thread budget, gates and protocol as the canonical
table above, plus the B02 regimes: E5 exact-truncation cases at 128/512
tokens, GPT-2 prefill at 1/32/128 tokens, and teacher-forced single-token
decode with past 1/32/128/512. Three fresh processes at commit `6932c42`
(no product-code differences from 0.2.0; `src/` identical to `4495fc6`).
Machine-readable artifacts with tick raws, `casedef`/`warmup` records and
per-rep prefill gates live in
`tests/Lokad.Onnx.Bench/baseline/summary-20260913-b02.json`, regenerated by
`python eng/parse_baseline.py <rep logs>` (extended rows render only when
present-and-ok in every rep). Confinement ratios were 0.94, 0.91 and 0.85:
no warnings, same background-load band. The frozen five reproduce in these
shared reps (30tok 1.5x, dinov3 1.3x, gpt2-4tok 1.2–1.3x, 8tok 2.8–3.1x,
resnet 2.4–2.6x), so future revisions compare within these reps.

| Case | rep1 L/ORT ms | rep2 L/ORT ms | rep3 L/ORT ms | Lokad / ORT |
|---|---:|---:|---:|---|
| e5-128tok | 71.4 / 46.8 | 68.0 / 46.0 | 69.9 / 46.5 | 1.5-1.5x |
| e5-512tok | 453.7 / 230.9 | 431.2 / 227.8 | 490.2 / 239.7 | 1.9-2.0x |
| gpt2-1tok | 38.2 / 27.6 | 39.9 / 31.1 | 45.6 / 31.5 | 1.3-1.4x |
| gpt2-32tok | 147.7 / 93.5 | 135.4 / 91.9 | 122.5 / 78.2 | 1.5-1.6x |
| gpt2-128tok | 467.2 / 257.6 | 445.8 / 274.0 | 415.2 / 246.2 | 1.6-1.8x |
| gpt2-dec-p1 | 35.9 / 26.9 | 40.4 / 31.2 | 35.8 / 27.7 | 1.3-1.3x |
| gpt2-dec-p32 | 45.1 / 28.8 | 41.0 / 27.4 | 39.9 / 27.4 | 1.5-1.6x |
| gpt2-dec-p128 | 49.6 / 28.4 | 51.8 / 29.0 | 49.8 / 28.6 | 1.7-1.8x |
| gpt2-dec-p512 | 97.8 / 32.4 | 99.8 / 31.6 | 101.1 / 32.6 | 3.0-3.2x |
| e5-30pad128 | 84.6 / 48.3 | 77.3 / 46.2 | 123.9 / 64.6 | 1.7-1.9x |

Two findings for the optimization epics. First, the E5 gap grows with
length (1.5x at 30–128 tokens, 1.9–2.0x at 512): the A02 tiled-attention
case. Second, the decode gap grows with past length while ORT's decode step
stays ~flat (27–33 ms from past 1 to past 512): Lokad re-pays
past-proportional work per step (36 → 40 → 50 → 100 ms), ORT does not —
the A03 KV-traffic case, with past-512 the highest-leverage decode target.
New GPT-2 regimes use a non-repeating token stride: the legacy 4-cycle trips
a narrow 1.5e-4 operating-point breach at 32+ tokens (identical situs in all
five probes, position 23; stride gates at 4.6e-5 through 512), recorded as
N01 material. Frozen `gpt2-4tok` keeps the legacy cycle byte-identical.

The padded case (30 real tokens + padding to 128) costs more than dense-128 on the Lokad side in all three reps (+13%, +14%, +59% on elevated rep3) while ORT is padding-indifferent (46–48 vs 46–47 ms on reps 1–2): mask-application or unskipped padded compute, priced for F02/G02 by B03 attribution before anyone "fixes" it. Rep3 of the pad campaign ran under renewed box load (both engines elevated); its ratios hold, so the rep stands. Pad artifacts: `tests/Lokad.Onnx.Bench/baseline/summary-20260913-pad.json` (e5-only campaign, `--cases` subset flag added to the parser for partial campaigns).
## Historical methodology

Each model case reports three validated rows measured in one process:

- `defaults` — Lokad `ExecutionOptions.Default` against a default ORT
  `InferenceSession` (default thread pool, `ORT_ENABLE_ALL` graph
  optimizations).
- `one-thread` — Lokad Scalar mode (`TensorExecutionOptions.Scalar`,
  `MaxDegreeOfParallelism=1`) against ORT with `IntraOpNumThreads=1`,
  `InterOpNumThreads=1`, `ORT_SEQUENTIAL`.
- `mode-threads=N` — Lokad `--mode` with `MaxDegreeOfParallelism=N`
  against ORT with `IntraOpNumThreads=N`, `InterOpNumThreads=1`,
  `ORT_SEQUENTIAL`.

With `--threads 1`, the second and third rows share the same thread budget
but differ in kernel mode (Scalar vs the selected `--mode`, here Auto with
vectorization enabled and one thread). Keeping both documents that
distinction instead of implying a new thread condition.

Per-row isolation: each row opens its own ORT session, validates, times,
then disposes it before the next row starts, so worker threads and
optimization state from one condition do not leak into the next. The Lokad
`ComputationalGraph` is loaded once per model and `Reset()` between
executions; ORT sessions are the isolated unit. Engine order alternates per
iteration (ORT-first on even iterations, Lokad-first on odd iterations) to
avoid giving either engine a systematic warm-cache advantage.

Validation before timing: every row validates first, outside timing, with
`BenchValidate.RequireAgreement` — exact output names, float32 dtype, exact
shapes, finite values, and per-element relative tolerance 1e-4 on every
declared output against the actual session used for that row. A diverging
row throws instead of timing incorrect results.

Timing boundaries per timed iteration: the measured region is exactly one
`graph.Execute` (Lokad) or one `session.Run` (ORT) with all declared outputs
requested. Outside the region: `Reset` (reported separately as `reset`),
input conversion to `OrtValue`s (reported as `convert`), validation
executions plus comparison (reported as `validation`), and ORT output
disposal (`disposal=outside`; `using` scopes end after the stopwatch stops).
Tokenization and model parsing happen once before timing. Three warmup
executions per engine precede nine timed iterations; the tables report best
and median with p95/max in the transcript, plus the raw samples below.

No process affinity or power plan was pinned; logical processor count only,
worker spinning left at ORT defaults. Equal thread budgets do not imply
equal CPU utilization between the two engines.

## Environment — 2026-09-08 run

Runner transcript host line:

    host=LOKAD-0399 cpu=Intel64 Family 6 Model 183 Stepping 1, GenuineIntel procs=28 (logical, no affinity pinning) fma=True runtime=.NET 10.0.11 lokad=0.2.0.0 ort=1.23.2.0 ort-provider=cpu-only ort-optimizations=ORT_ENABLE_ALL mode=auto threads=1 iters=9 warmup=3

- Host: LOKAD-0399, Intel Core i7-14700KF, 20 cores / 28 logical processors.
- Build: Release, .NET SDK 10.0.300-preview.0.26177.108, .NET runtime 10.0.11.
- Source: commit `b1b447d` plus the benchmark-harness rework in this commit
  (`tests/Lokad.Onnx.Bench/Program.cs`: per-row sessions, alternated order,
  raw samples, `validation`/`convert` boundaries, output shapes, sidecars).
  Rebuilding at this commit and rerunning the command below reproduces the
  harness; numbers remain machine- and load-dependent.
- Native reference: `Microsoft.ML.OnnxRuntime` 1.23.2.0, CPU provider only,
  graph optimizations `ORT_ENABLE_ALL`, default worker spinning, no affinity.

## Assets

All models and sidecars are local; parsing and tokenization occur outside
timing. Full hashes live in
[ModelManifest.json](tests/Lokad.Onnx.Backend.Tests/ModelManifest.json).

| Asset | Model bytes | SHA-256 prefix | Sidecar | Inputs | Outputs |
|---|---:|---|---|---|---|
| multilingual-e5-small | 470,268,510 | CA456C06B3A9 | tokenizer `sentencepiece.bpe.model`, 5,069,051 bytes, CFC8146ABE2A | three int64 tensors, 1x8 or 1x30 (`input_ids`, `attention_mask`, `token_type_ids`) | `last_hidden_state` 1x8x384 / 1x30x384 |
| DINOv3 ViT-S/16 (full weights) | 137,969 | BB75E9E30FF3 | `model.onnx_data`, 86,347,776 bytes, 1EFF0BB9F4FD | float32 1x3x224x224 (`pixel_values`, filled with 0.5) | `last_hidden_state` 1x201x384, `pooler_output` 1x384 |
| ResNet50 feature export | 93,961,728 | 4F0558B775C8 | none | float32 1x3x224x224 (`input`, filled with 0.5) | `output` 1x2048 |
| GPT-2 (past-state) | 498,126,358 | 42C1E92A21C4 | none | `input_ids` 1x4, `attention_mask` 1x4, `position_ids` 1x4, 24 empty past tensors 1x12x0x64 | `logits` 1x4x50257 plus 24 present tensors 1x12x4x64 |

DINOv3 now runs against the full-weight asset (graph plus `model.onnx_data`)
and validates end to end; the earlier placeholder-asset caveat no longer
applies. DINOv2 is excluded by a tracked known-divergence condition (PLAN.md
C01, registry in `tests/Lokad.Onnx.Bench/KnownDivergences.cs`): after bit-identical
GELU fusion and tail order parity, `last_hidden_state` still diverges at reference-scaled
1.92E-004 from uniform depth-amplified fp32 summation-order drift with no localizable
kernel defect, above the unchanged 1e-4 gate. A 2026-09-12 per-layer probe (temp instrumented copy, ORT 1.29, gate metric |ref-cand|/(1+|ref|)) shows embeddings agreeing at 8e-8, the layer-0 norm output already at 1.2e-6 in ORT-vs-ORT as well, and a layer-11 attention jump to 1.48e-4 at the identical token and channel in ORT-vs-ORT and in Lokad-vs-ORT; a 1-ulp input perturbation alone moves ORT's own output by 1.0e-4 (plain) and 1.2e-4 (fused). Cross-implementation 1e-4 agreement is therefore unachievable on this operating point, while DINOv3 passes because it runs native LayerNormalization/Gelu single ops with no fusion-order differences. The case validates, reports
`case-status dinov2-224=excluded-known-divergence`, and skips every timed row, so no
DINOv2 rows are published. A breach at or above the 1e-3 tripwire, or on any
unregistered case, still fails the run as a fresh regression.

## Historical results — 2026-09-08 methodology

`Bench e5 resnet50 dinov3 gpt2 --mode auto --threads 1 --iters 9`
(three warmups, nine timed iterations per row). Median ratio is
Lokad median / ORT median. `maxdiff` is the worst validated output of that
row. Raw samples follow the tables.

| Case | Row | Lokad best | Lokad median | ORT best | ORT median | Median ratio | maxdiff |
|---|---|---:|---:|---:|---:|---:|---|
| e5, 8 tokens | defaults | 167.6 ms | 203.0 ms | 3.1 ms | 3.4 ms | 59.7x | 7.28E-007 |
| e5, 8 tokens | one-thread | 190.5 ms | 232.4 ms | 5.6 ms | 11.6 ms | 20.0x | 9.52E-007 |
| e5, 8 tokens | mode-threads=1 | 116.8 ms | 126.2 ms | 5.5 ms | 5.8 ms | 21.8x | 7.28E-007 |
| e5, 30 tokens | defaults | 201.2 ms | 260.6 ms | 4.1 ms | 5.3 ms | 49.2x | 1.15E-006 |
| e5, 30 tokens | one-thread | 405.2 ms | 430.9 ms | 12.2 ms | 12.8 ms | 33.7x | 1.24E-006 |
| e5, 30 tokens | mode-threads=1 | 144.5 ms | 154.5 ms | 11.6 ms | 13.1 ms | 11.8x | 1.15E-006 |
| DINOv3, 224x224 | defaults | 333.0 ms | 445.7 ms | 18.6 ms | 20.4 ms | 21.8x | 5.99E-006 |
| DINOv3, 224x224 | one-thread | 2193.9 ms | 2281.5 ms | 69.9 ms | 72.8 ms | 31.3x | 4.87E-006 |
| DINOv3, 224x224 | mode-threads=1 | 335.9 ms | 384.8 ms | 72.0 ms | 80.6 ms | 4.8x | 5.99E-006 |
| ResNet50, 224x224 | defaults | 245.5 ms | 281.9 ms | 7.2 ms | 8.0 ms | 35.2x | 1.98E-006 |
| ResNet50, 224x224 | one-thread | 1776.8 ms | 1788.6 ms | 50.3 ms | 51.0 ms | 35.1x | 2.53E-006 |
| ResNet50, 224x224 | mode-threads=1 | 239.0 ms | 243.7 ms | 50.5 ms | 51.0 ms | 4.8x | 1.98E-006 |
| GPT-2, 4 tokens | defaults | 745.8 ms | 990.2 ms | 8.3 ms | 9.5 ms | 104.2x | 7.35E-006 |
| GPT-2, 4 tokens | one-thread | 952.0 ms | 1010.4 ms | 23.1 ms | 27.7 ms | 36.5x | 7.95E-006 |
| GPT-2, 4 tokens | mode-threads=1 | 783.8 ms | 936.0 ms | 24.6 ms | 30.5 ms | 30.7x | 7.35E-006 |

These historical timings show gaps under each recorded condition, but the
default ratios do not measure a gap with equal CPU resources. The matched
Auto rows suggest remaining single-thread work; affinity-controlled repeat
runs are required to quantify it. The Scalar one-thread row
is much slower than the Auto one-thread row on vision models (ResNet50
1788.6 ms vs 243.7 ms median; DINOv3 2281.5 ms vs 384.8 ms), which shows the
mode distinction carries the effect, not just the thread count. Profiling
attributes ResNet50 to Conv at 90.8% (lowered through the shared GEMM
dispatcher) and e5-small to MatMul at 76.0%; generalizing the unrolled FMA
kernel to K%32 != 0 shapes moved the ResNet50 matched row from 417.6 ms to
243.7 ms with identical validation diffs.

### Raw samples (ms, n=9 per engine per row)

    e5-8tok defaults:      lok=[210.65,196.39,187.63,208.51,249.59,197.03,203.02,207.92,167.58] ort=[3.07,3.59,3.39,3.44,3.19,4.17,3.58,4.50,3.39]
    e5-8tok one-thread:    lok=[234.72,232.40,270.13,190.55,238.79,213.15,195.85,246.91,219.87] ort=[15.47,11.81,12.57,6.04,6.65,13.05,11.58,8.88,5.61]
    e5-8tok matched:       lok=[120.55,116.75,136.72,127.05,118.00,129.66,121.10,128.43,126.23] ort=[6.66,5.80,5.54,5.88,5.84,6.58,5.74,5.83,5.62]
    e5-30tok defaults:     lok=[260.62,220.43,223.90,284.85,282.43,262.30,279.52,201.18,227.03] ort=[4.75,6.19,4.69,5.71,5.29,5.74,5.00,8.55,4.10]
    e5-30tok one-thread:   lok=[570.73,553.08,475.02,421.18,419.25,407.91,454.78,430.92,405.25] ort=[27.84,16.45,13.24,12.76,12.40,12.18,13.13,12.33,12.79]
    e5-30tok matched:      lok=[145.28,148.05,160.54,146.74,144.46,155.81,154.69,163.89,154.52] ort=[13.07,35.92,12.53,12.67,11.66,13.14,11.60,14.21,14.51]
    dinov3-224 defaults:   lok=[474.40,390.33,539.31,406.05,464.43,347.45,445.66,332.98,462.18] ort=[18.98,36.59,20.41,22.00,20.04,21.29,18.84,28.46,18.63]
    dinov3-224 one-thread: lok=[2254.65,2254.56,2495.61,2430.75,2193.92,2249.55,2333.49,2281.47,2286.12] ort=[74.72,117.84,93.44,70.64,69.92,78.86,72.76,72.00,71.75]
    dinov3-224 matched:    lok=[387.11,384.81,411.62,337.38,348.35,395.20,335.89,390.08,357.03] ort=[96.41,73.04,80.70,72.02,80.56,78.83,72.95,99.71,88.74]
    resnet50 defaults:     lok=[281.87,270.54,289.81,245.50,290.84,257.56,342.69,248.34,359.53] ort=[7.22,21.58,7.43,23.46,7.50,11.27,8.01,8.11,7.69]
    resnet50 one-thread:   lok=[1793.93,1827.50,1798.57,1776.91,1814.86,1776.76,1785.98,1778.50,1788.61] ort=[51.08,50.69,50.33,51.04,51.38,50.63,51.09,51.08,50.75]
    resnet50 matched:      lok=[243.70,248.59,242.20,242.75,240.80,245.79,267.52,261.21,239.03] ort=[50.81,50.84,51.04,54.32,51.15,51.43,50.88,51.49,50.51]
    gpt2-4tok defaults:    lok=[995.01,1060.91,990.17,1363.00,1061.80,988.22,864.71,745.80,941.57] ort=[9.59,9.48,8.64,9.35,10.91,13.20,8.33,10.77,9.10]
    gpt2-4tok one-thread:  lok=[1090.96,1145.03,1066.73,984.21,1009.41,1069.98,1010.40,986.62,952.05] ort=[26.59,31.09,23.10,24.91,36.88,55.57,27.69,27.90,24.14]
    gpt2-4tok matched:     lok=[881.01,858.79,783.81,843.33,1053.93,943.16,935.97,1059.81,1376.17] ort=[48.68,37.50,27.22,27.40,24.61,30.45,26.38,38.58,37.08]

Per-row support values from the transcript: `reset` best 0.0 ms on every
row (in-place reset cost is negligible next to inference); `convert` 0.0 ms
on e5/GPT-2 rows and 0.1–0.2 ms on vision rows; `validation` 160.8–2350.8 ms
(one full Lokad execute plus one full ORT run plus comparison per output,
outside timing); `disposal=outside` throughout.

## Limits of this comparison

- One machine, one process per model set, nine samples per row. These samples
  do not establish a stable single-CPU baseline or a release performance gate.
- Running `Bench all` (or naming `dinov2`) reports the DINOv2 exclusion above
  by design and continues the remaining cases; publish tables only for validating
  models and keep the exclusion stated.
- The Lokad graph is reused across the three rows of a model with `Reset`
  between executions while ORT sessions are per-row; residual pool/cache
  effects on the Lokad side across rows are not measured separately.
- The old latency/microbenchmark tables are available in Git history; they
  predate substantial kernel and allocation changes and should not be reused
  as current measurements.

## Reproduce the historical setup

From the repository root, using the existing local assets. This command
reproduces the three-condition setup; it does not enforce the replacement
single-CPU contract described above:

```powershell
dotnet build Lokad.Onnx.slnx -c Release --tl:off --nologo -v minimal
dotnet tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll e5 resnet50 dinov3 gpt2 --mode auto --threads 1 --iters 9
```

The runner prints the host line, one `case` line per row (asset identity,
input/output shapes, warmup, iterations), one result line per row (best,
median, p95, max, reset, convert, validation, disposal, maxdiff), and one
raw-sample line per row. Keep the default, one-thread and equal-budget
results distinct; equal limits do not imply equal CPU utilization. Validate
every named output outside timing for each actual session.

For operator benchmarks, the current command is `dotnet tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll micro ops`;
`matmul2d`, `matmul` and `indexing` cover other kernel cases. The `micro oneop` lane compares five frozen one-op models
(tests/Lokad.Onnx.Bench/oneop) between a Lokad graph and a single-CPU ORT session on identical inputs, gating timing on
1e-4 agreement. Pin execution modes and record
allocations as well as latency; profiler-enabled timings are separate.

`bench.ps1` is a startup-inclusive CLI benchmark by design: it launches a
fresh CLI process for every e5 sample, so its warmup process cannot warm
those subsequent JITs/sessions. Read its `graphMs`/`wallMs` series as
per-process startup plus inference, not warmed inference throughput.
Persistent-process inference timing lives in the Bench runner above.
