# Matched audio application comparison

This lane measures complete PCM requests through Lokad.Onnx and Microsoft
ONNX Runtime. It covers Parakeet TDT 0.6B V3 transcription and pyannote
Community-1 diarization. Both engines receive the same decoded mono 16 kHz
FP32 audio and must reproduce the retained application results on every call.

The [measured Parakeet and pyannote results](results-20260919.md) include explicit
Microsoft ORT times. The separate [Whisper comparison](../whisper-comparison/results-20260919.md)
uses the same twenty ASR clips. [BENCHMARK.md](../../../BENCHMARK.md) summarizes
all three comparisons in its first table.

`Program.cs` times `ParakeetTranscriber.Transcribe` or
`Community1Diarizer.Diarize`. `native_adapters.py` runs the corresponding
graphs through ORT and implements the same request policy using pinned upstream
methods. Both include frontend computation, inference, decoding or clustering,
timeline reconstruction, and construction of owned results. Loading, input
file reads, hashes, serialization and external validation are outside the timer.

The native Parakeet adapter runs frontend, encoder and decoder/joint ONNX
graphs. It starts fresh recurrent states for every request. Its greedy loop
matches the independently crosschecked reference generator, without the
generator's duplicate decoder execution or intermediate exports.

The native pyannote adapter runs segmentation, embedding encoder and projection
ONNX graphs. Pinned Torch/torchaudio methods compute features and weighted
pooling; NumPy/SciPy and upstream pyannote methods perform PLDA/VBx clustering
and reconstruction. Upstream capture helpers supply window and mask policies;
their bookkeeping is included. The comparison is therefore an **ORT-backed
application baseline**, not an ORT-kernel-only benchmark or the unmodified
upstream Python pipeline. Segmentation and embedding encoder each run once per
window, with three speaker projections. Both applications use stable speaker
tie ordering and automatic speaker counts.

The fixed workloads are all twenty labeled ASR clips (213.265 seconds in total)
and one annotated thirty-second dialogue with its three ten-second crops.
The crops are correlated with the full dialogue. No maximum-duration or general
recognition/diarization accuracy claim follows from this timing lane.

## Protocol

The Windows supervisor runs on CPU 0. Workers inherit affinity to logical CPU 2
before Python or .NET starts; the supervisor checks actual process and descendant
affinity throughout. ORT uses CPUExecutionProvider, one intra-op and inter-op
thread, sequential execution, all graph optimizations, and disabled spinning.
Torch and BLAS thread budgets are one. Managed requests use the public API's
normal options, with no LOKAD, DOTNET or COMPlus overrides.

Conformance runs one complete corpus for each engine and family. Only after all
four workers succeed does timing begin. The fixed order for each family is
managed, ORT, ORT, managed. Each fresh timing process performs one full corpus
warmup and three full measured passes. This gives 120 measured Parakeet requests
per engine and 24 measured pyannote requests per engine, with all samples retained.
Two process visits describe observed variation; they do not establish confidence
bounds or calibrated parity. Process snapshots record foreign activity but do not
prove an exclusive or idle Windows machine.

Every request checks exact Parakeet text, tokens, frames, duration choices and
stop metadata. Pyannote discrete decisions agree exactly, interval endpoints use
the existing 1e-12 bound, and all centroid components use the existing scaled
1e-4 gate. Inputs and held outputs must remain unchanged, and repeated results
within each worker must match. Known intermediate numerical failures in the
broader model qualifications remain failures.

## Reproduce using the retained local assets

The preparation script verifies pinned models, upstream sources and closed
reference artifacts. Those large assets are git-ignored; prepare them through
the existing [ASR](../accuracy/README.md) and
[dialogue](../../pyannote/dialogue/README.md) lanes first. This lane does not
download assets or regenerate old reference receipts. Python 3.13 uses NumPy
2.2.4, ORT 1.29.0, Torch/torchaudio 2.11.0+cpu, SciPy 1.16.3, einops 0.8.1,
pyannote.core 6.0.1 and sortedcontainers 2.4.0. The supervisor needs psutil 7.0.0.
`supervise.py` names the existing interpreter and dependency directories.

Use a fresh artifact directory and the already qualified product DLL directory:

```powershell
python -X utf8 tests/audio/comparison/prepare.py --output artifacts/audio-comparison-new/inputs
dotnet build tests/audio/comparison/AudioBenchmark.csproj -c Release -o artifacts/audio-comparison-new/bin -p:FrozenProductDirectory=C:/Users/JoannesVermorel/code/Onnx/artifacts/whisper-recording-v2-20260919/recording-bin --tl:off --nologo -v minimal
artifacts/asr-labeled-20260919/venv/Scripts/python.exe -X utf8 tests/audio/comparison/supervise.py --artifact artifacts/audio-comparison-new --mode conformance --name conformance
artifacts/asr-labeled-20260919/venv/Scripts/python.exe -X utf8 tests/audio/comparison/supervise.py --artifact artifacts/audio-comparison-new --mode timing --name timing
python -X utf8 tests/audio/comparison/audit.py --artifact artifacts/audio-comparison-new --output artifacts/audio-comparison-new/summary.json
```

Constructor timings are separate observations: the managed API constructor and
native adapter constructor have different import/setup boundaries. They exclude
prior asset verification and are not cold-start comparisons. Peak RSS is sampled
across each complete process, including verification, construction and all calls;
it is not a per-request allocation or a memory ceiling.

The frozen product is source `8732831b52a97b009ab3edbd5319a56269e19449`;
execution code matches `c6bf781`. The supervisor refuses other Core/Data DLL
hashes or different conformance/timing payloads. Preserve failed attempts and use
a new directory after fixing a failure. Do not edit captured source, binary or
input files during an active campaign. The auditor checks the recorded source
files; extra files added after a campaign are not part of its frozen payload.
