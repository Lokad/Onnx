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
and resamples to 16 kHz. Default requests are limited to 30 seconds. The separate
[`--recording` mode](../recording/README.md) accepts up to ten minutes in independent
windows, exposing quiet/hard boundaries and partial progress. Parakeet recognizes
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

On the recorded Windows run, full numerical qualification remains **failed**.
All 784 saved output arrays
(3,090,494 values) are compared, including every recurrent state. Three arrays
exceed the unchanged `abs(actual-reference)/max(1,abs(reference)) <= 1e-4` gate.
They reproduce the same duration-zero logit at English decoder call 26 across
the normal, one-token-per-frame and repeat cases; maximum error is
`2.321004867553711e-4`. All application decisions still agree. Native ORT given
managed encoder output reproduces almost all of the discrepancy, localizing it
to propagated frontend/encoder differences. This evidence does not establish
general speech accuracy, long-audio behavior or INT8 support. The later complete
AMD qualification is recorded below. The initial local process peak near 6.3 GB
is a finite observation.

A separate [twenty-recording labeled check](../../audio/accuracy/results-20260919.md)
matches all native application decisions and measures 11 word errors in 559
reference words (1.9678%) on clean English speech. It leaves these numerical
failures and broader multilingual/long-recording accuracy open.

## Reproduce the opt-in full-array replay

Install the pinned packages in `../requirements.txt`. Supply the recorded PCM
manifest from the [shared recorded-audio recipe](../../audio/README.md) and a checkout of
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


## Complete AMD pipeline qualification

The SDK 10.0.204 source archive at
`c6bf781816f3a45260f7a5eb33fb05d640a821e3` also passes the complete replay on
AMD EPYC 9V74, runtime 10.0.8, confined to CPU 2. Both the nine enabled defaults
and their exact-`0` comparison settings pass all seven application cases and
six invalid/canceled-request recovery checks per process. Every engine carries
its own recurrent states; caller inputs and retained tensors/results are checked.

Both independent NumPy audits pass **all 784 arrays / 3,090,494 values** against
the same reproduced native reference. Maximum scaled error is
`3.830720152728115e-5`, with no failed arrays at the unchanged `1e-4` gate.
All 784 arrays match bit for bit between the two AMD configurations. Application
decisions also match Windows; 202 of 784 arrays match bits across hosts.

This AMD pass does not erase the three recorded Windows duration-logit failures
above. Their platform difference has not been attributed to one instruction or
kernel. No arithmetic, model file, fixture or tolerance was changed for this
replay. Broader accuracy and long-recording acceptance remain open.

Sampled process-group peaks are 5,784,256,512 bytes with the nine mechanisms
disabled and 6,712,479,744 bytes with defaults enabled. Both stay below the
13 GiB guard on the 16 GiB/no-swap VM, with normal runtime settings and no forced
GC. These are finite short-recording observations; the higher enabled peak is
retained. Total supervised job durations of about 40 seconds include model
verification/loading and evidence generation, and are not inference benchmarks.

The six models were transferred individually and verified before reuse.
The first launch stopped before inference because its assumed runner dependency
was absent; that failed attempt remains archived. The corrected immutable
payload includes all runner dependencies and both jobs completed before
collection. The independent audit verifies the full schedule, binary/asset
identities, actual affinity, resources and saved arrays.

Ignored local evidence is under `artifacts/parakeet-amd-v2-20260919`, with
`summary.json`, per-configuration audits, raw collection and the frozen recipe.
Source/binary identities match the [CPU defaults qualification](../../e5/defaults-20260919.md).
SHA-256 identities:

- Reference manifest: `3bad7d262b8809b1265c84c8e66d02ee38e7d4cff2d92014448976a9e161103c`
- Frozen bundle: `bbbcd25017621df4f1dd8ca59dd1801fef5c46ed73d9d771cb56be04318003dc`
- Collected AMD results: `14e7513ff61d58565c2dcac15671377071e8f17a14471658af17863aaf2f53e9`
