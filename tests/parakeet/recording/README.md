# Parakeet recording transcription

`ParakeetTranscriber.TranscribeRecording` accepts finite mono 16 kHz PCM up to
ten minutes. The CLI enables it with `--model-type parakeet --recording`:

```powershell
lonnx.cmd transcribe models/parakeet-tdt-0.6b-v3 longer.wav --model-type parakeet --recording --json
```

```csharp
var model = new ParakeetTranscriber("models/parakeet-tdt-0.6b-v3");
var result = model.TranscribeRecording(monoPcm, 16000,
    ParakeetRecordingOptions.Default, CancellationToken.None);
Console.WriteLine(result.Text);
Console.WriteLine($"{result.StopReason}: {result.ProcessedSeconds}/{result.DurationSeconds} s");
```

Use the same six pinned local FP32 assets as the [short API](../transcribe/README.md).
The existing `Transcribe` method retains its thirty-second bound and behavior.
Recording mode uses managed inference and does not download assets or load ORT.
It serializes requests per instance, owns the input copy used by its windows,
and returns immutable collections. Cancellation takes effect between graph calls.

Each window is independent: frontend features, recurrent states and previous
token start afresh. A window contains at most thirty seconds. If more audio
remains, its final five seconds are scanned in 10 ms blocks. A block is quiet
when its mean squared amplitude is at most `0.000009` (RMS 0.003). The midpoint
of the longest quiet run of at least 200 ms becomes the boundary; equal runs
choose the later one. Otherwise the window ends exactly at thirty seconds.
The final window uses all remaining audio. A final nonsilent tail shorter than
257 samples is zero-padded to the frontend minimum for inference only.

This amplitude rule is not voice activity detection or word alignment. Quiet
speech can be cut, and continuous speech may reach a hard cut. No samples are
skipped or overlapped, and no text deduplication hides boundary errors. Windows
report `Quiet`, `HardLimit` or `EndOfRecording`; hard cuts can lose or duplicate
words. The public text joins completed nonempty window texts with one space.
Natural long-conversation accuracy requires separate qualification.

`Windows` retains each window's recording-relative start, actual audio duration
and complete `ParakeetTranscription`. Its token frame indices are local to that
window, nominally 80 ms per frame; predicted durations are not word boundaries.
They can extend beyond a tiny padded tail's actual audio duration.

Default work limits are 256 windows and 4096 tokens per window, with at most
ten emissions per encoder frame. Allowed window limits are 1–512; token limits
are 1–4096 and frame limits 1–10. The CLI's `--max-tokens` applies per window and
`--max-windows` requires `--recording`.

`Completed` means every sample's window finished decoding. `WindowLimit` returns
all completed windows. `TokenLimit` keeps the unfinished window under `Windows`
but excludes it from top-level `Text` and `ProcessedSeconds`. Inspect its
`Decoding.Text` when partial text is useful. Empty input completes with no windows;
exact digital silence completes without graph execution. CLI partial results
are valid output with exit code zero and an explicit stderr diagnostic.

## Qualification recipe

The [recorded results](results-20260919.md) distinguish application agreement,
fixed-label accuracy, resource observations and outstanding numerical gates.
These scripts do not replace the complete intermediate-array conformance lane.

Use the fixed labeled PCM recipe in [audio accuracy](../../audio/accuracy/README.md).
The two constructed recordings can be prepared using
[Whisper recording inputs](../../whisper/recording/README.md); the Parakeet
preparer reuses their bytes and labels without invoking Whisper. All source
utterances and hashes are retained. The additional hard-boundary input adds a
fixed 0.02 DC offset; the ten-minute input cyclically repeats the connected
recording. They are correlated diagnostics, not independent natural recordings.

From the repository root, with a fresh destination:

```powershell
dotnet build tests/parakeet/recording/RecordingReplay.csproj -c Release --tl:off --nologo -v minimal -o artifacts/parakeet-recording-new/bin
dotnet build src/Lokad.Onnx.CLI -c Release --tl:off --nologo -v minimal -o artifacts/parakeet-recording-new/cli-bin
python -X utf8 tests/parakeet/recording/prepare.py --source <constructed-inputs>/inputs.json --output artifacts/parakeet-recording-new/inputs
python -X utf8 tests/parakeet/recording/supervise.py --artifact artifacts/parakeet-recording-new --python <native-python.exe>
python -X utf8 tests/parakeet/recording/audit.py --artifact artifacts/parakeet-recording-new --output artifacts/parakeet-recording-new/audit.json
python -m unittest discover -s tests/parakeet/recording -p test_audit.py
```

Native Python needs NumPy 2.2.4 and ORT 1.29.0. The supervisor/auditor additionally
need psutil 7.0.0 and the fixed accuracy recipe's JiWER dependency. The supervisor
currently targets Windows CPU 2, runs four workers sequentially and enforces
20 GiB / 1800-second guards per worker. It freezes executable source and binaries
and records worker identity, affinity and RSS samples. Both .NET builds must
contain identical Core/Data DLLs. Build from the same source revision before
running; leave frozen executable source unchanged until the supervisor ends.

The native decoder carries its own state and checks its trajectory against the
original onnx-asr decoder, pinned in [assets.json](../transcribe/assets.json).
The independent auditor compares every window/token/frame/duration/stop decision,
validates coverage and source/model/binary identities, checks complete versus
partial progress, recomputes word errors, and checks worker termination/resource
records. It refuses existing output. Preserve failed runs and never overwrite
closed successful evidence.
