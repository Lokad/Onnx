# Whisper Large V3 Turbo: matched application timing

This additive lane supplies fresh Lokad.Onnx and Microsoft ORT timings for the
same twenty clean-English clips used by the Parakeet comparison: 213.265 seconds,
ten speakers. It keeps the completed Parakeet/pyannote tools and evidence unchanged.

The timed request starts with decoded mono 16 kHz FP32 PCM and includes feature
extraction, encoder inference, cached greedy decoding, no-speech decisions, text
decoding and owned result construction. Model loading, file reads, hashes,
serialization and external checks are outside the timer. Native execution uses
the pinned Transformers NumPy frontend plus three CPU ORT graphs; this is an
ORT-backed application baseline, not isolated ORT kernel timing.

Both engines use English transcription, no timestamps and at most 444 new
tokens. Every request starts fresh decoder caches. Each native request computes
the full frontend again; its output is checked byte-for-byte against the retained
feature fixture **after** the timer. No precomputed feature is used as inference
input. Text, every token ID, stop reason and no-speech skip decision must agree
with the closed independent reference. Inputs and retained results must survive
all later requests unchanged.

Two fresh conformance processes run ORT then managed over the complete corpus.
An independent successful conformance audit is required before the four timing
workers run in managed → ORT → ORT → managed order. Each timing process performs
one full warmup pass and three measured passes: 120 measured requests per engine,
240 total. Every timing sample remains, including GC and slow calls.

Workers inherit logical CPU 2 before runtime startup; the supervisor uses CPU 0.
ORT has one intra/inter-op thread, sequential execution, all graph optimizations
and no spinning. BLAS budgets are one and tokenizer parallelism is disabled.
Managed calls use normal public options with no LOKAD/DOTNET/COMPlus override.
The process group is bounded to 20 GiB sampled RSS and 3,600 seconds per worker.
Foreign CPU activity and both process visits are reported. The Windows host is
an active workstation; these observations cannot establish calibrated confidence
or AMD parity.

The product is the same qualified archive-built source as the earlier audio
baseline, `8732831b52a97b009ab3edbd5319a56269e19449`; its short Whisper core
matches `c6bf781`. Core DLL
`05884cfd524cc7130321f5dc1bcd0af17dddc7b97e8428d2d2f59e00edb795c2`, Data DLL
`27598aa8d8c6b97a1415302cf3aaced1adcf53b20b64734c0c0e047492ca069d`.
This lane does not change product arithmetic or the known full encoder/logit
numerical failures. The previous 10/559 word-error observation is unchanged;
new timings do not add an accuracy corpus or maximum-duration qualification.

## Reproduction

Use the existing models and closed ASR references from
[the labeled accuracy lane](../accuracy/README.md). `prepare.py` verifies
streaming hashes, including the greater-than-2-GB external encoder data. It
downloads nothing. Python 3.13 dependencies are NumPy 2.2.4, ORT 1.29.0,
Transformers 5.16.1 and tokenizers 0.23.2. The supervisor needs psutil 7.0.0.

From the repository root, using a fresh artifact directory:

```powershell
python -X utf8 tests/audio/whisper-comparison/prepare.py --output artifacts/whisper-ort-new/inputs
dotnet build tests/audio/whisper-comparison/WhisperBenchmark.csproj -c Release -o artifacts/whisper-ort-new/bin -p:FrozenProductDirectory=C:/Users/JoannesVermorel/code/Onnx/artifacts/whisper-recording-v2-20260919/recording-bin --tl:off --nologo -v minimal
python -m unittest discover -s tests/audio/whisper-comparison -p test_protocol.py
artifacts/asr-labeled-20260919/venv/Scripts/python.exe -X utf8 tests/audio/whisper-comparison/supervise.py --artifact artifacts/whisper-ort-new --mode conformance --name conformance
python tests/audio/whisper-comparison/audit.py --artifact artifacts/whisper-ort-new --phase conformance --output artifacts/whisper-ort-new/conformance-audit.json
artifacts/asr-labeled-20260919/venv/Scripts/python.exe -X utf8 tests/audio/whisper-comparison/supervise.py --artifact artifacts/whisper-ort-new --mode timing --name timing
python tests/audio/whisper-comparison/audit.py --artifact artifacts/whisper-ort-new --phase timing --output artifacts/whisper-ort-new/summary.json
```

The supervisor binds the local prospective plan
`.agent/m4-whisper-ort-baseline-20260919.md` before execution. It freezes every
source, dependency, input manifest and DLL and refuses changes between phases.
Do not edit those files during a campaign. Observe existing process handles
after timeouts; never launch a duplicate worker. Preserve any failed attempt in
its original directory. New writers refuse existing output.
