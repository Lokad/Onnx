# Multilingual and controlled-noise ASR check

This lane compares complete public Parakeet and Whisper transcription with
independent Microsoft ORT applications on a fixed labeled sample. Recognition
errors against human transcripts are separate from native/managed agreement.
It does not replace full numerical qualification or the existing repeated
audio performance comparison.

The data is Google's [FLEURS](https://huggingface.co/datasets/google/fleurs),
revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, licensed CC-BY-4.0.
See Conneau et al., [FLEURS: Few-shot Learning Evaluation of Universal
Representations of Speech](https://arxiv.org/abs/2205.12446), 2022.
[dataset.json](dataset.json) binds the five original test parquet files and card
to their complete hashes, verified against the Hub's immutable revision/etags.
The sample includes English, French, German, Spanish and Italian. These are read
sentences, sometimes parallel translations, and are not conversational speech.
Model training exposure is not established by the dataset's test-split label.

For each language, select four distinct sentence IDs without recognition:
two rounds through the sorted available gender labels, each selecting the first
numeric ID/filename in the inclusive 4–10 and 12–25 second bands. Previously
chosen IDs are excluded. The German test split has only gender label 0, so both
rounds use that label. An initial preparation that required both labels failed
before audio transforms or inference; that failure is retained separately.
The full candidate metadata and corrected selection are frozen before inference.
Gender labels do not establish speaker identities or population coverage.

SoundFile and ffmpeg must decode each selected recording to identical mono
16 kHz float32 samples. Retain the original file and PCM. Each source has a
clean and a deterministic additive-noise counterpart: PCG64 normal noise,
mean removed, scaled to 10 dB using power over the whole recording. The seed is
the first eight little-endian SHA256 bytes of `revision/locale/filename` in UTF-8.
A common gain, at most one, keeps both versions below a 0.99 peak before float32
rounding. The auditor reconstructs the signals and independently verifies actual
SNR within 0.001 dB. This tests artificial broadband noise.

Score the raw human transcription using Unicode NFKC and casefold, canonical
typographic apostrophes, retained letters/digits/internal apostrophes, other
punctuation replaced by spaces, and collapsed whitespace. Accents remain;
numbers are not verbalized. Independent edit distance must agree with JiWER.
Corpus WER/CER use total errors divided by total reference units, with spaces
included in CER. Clean/noisy and language totals remain separate. Every error
and transcript is retained; no accuracy threshold is fitted to the outputs.

Four fresh workers run sequentially: Parakeet ORT, Parakeet managed, Whisper
ORT, Whisper managed. Each executes forty cases plus a repeat of the first,
for 164 complete requests. Both engines start independent request states and
recompute features from PCM. Whisper receives the declared language; Parakeet
detects language automatically. Native results never feed managed inference.
Input and held-output ownership are checked throughout. The auditor compares
all public decisions and independently scores both engines against human labels.

Windows workers inherit CPU 2 before runtime startup; supervision uses CPU 0.
ORT 1.29.0 uses one intra/inter-op thread, sequential execution and all graph
optimizations. .NET uses its ordinary runtime and qualified defaults. Each worker
has a 20 GiB RSS / 3,600-second guard and requires at least 1 GiB system available
memory throughout; managed preflight requires 20 GiB available. Memory preflights
are retained; the supervisor may wait up to ten minutes before creating a managed
worker, then fails if the threshold remains unmet. All sampled
process births, affinity, memory and foreign activity remain. Per-request times
are single-pass observations, without performance confidence claims.

Use the existing dataset/scoring environment and native interpreter. Required
versions are those in [the earlier accuracy lane](../accuracy/requirements.txt),
plus psutil 7.0.0. The retained Windows supervisor starts native workers with
`C:/Python313/python.exe`; that interpreter must have the pinned native packages.
The scoring environment also supplies PyArrow, SoundFile, JiWER and psutil.
Preparation requires ffmpeg on PATH. The model/data pins are mandatory; no new
model download is needed in this checkout. From the repository root, choose a
new artifact path and restore the qualified product binaries before these steps:

```powershell
$scorePython = 'artifacts/asr-labeled-20260919/venv/Scripts/python.exe'
$nativePython = 'C:/Python313/python.exe'
$artifact = 'artifacts/asr-multilingual-new'
$product = (Resolve-Path 'artifacts/softmax-zero-product-20260919/frozen').Path
& $scorePython -X utf8 -B tests/audio/multilingual/prepare.py --dataset models/fleurs-accuracy --output "$artifact/inputs"
& $scorePython -X utf8 -B tests/audio/multilingual/audit_inputs.py --dataset models/fleurs-accuracy --inputs "$artifact/inputs" --output "$artifact/input-audit.json"
dotnet build tests/audio/multilingual/MultilingualReplay.csproj -c Release -o "$artifact/bin" "-p:FrozenProductDirectory=$product" --tl:off --nologo -v minimal
& $scorePython -X utf8 -B -m unittest discover -s tests/audio/multilingual -p 'test_*.py' -v
& $nativePython -X utf8 -B tests/audio/multilingual/prepare_runtime.py --artifact $artifact
& $scorePython -X utf8 -B tests/audio/multilingual/supervise.py --artifact $artifact
# Only after the supervisor and all recorded child births are absent:
& $scorePython -X utf8 -B "$artifact/runtime-source/audit.py" --artifact $artifact --output "$artifact/audit.json"
& $scorePython -X utf8 -B tests/audio/multilingual/check_results.py --artifact $artifact --output "$artifact/record-checks.json"
```

The replay binds qualified production source `087e280`, core `187de61a` and
Data `809242b5`; the runtime manifest verifies all existing model files and native
sources. Successful writers refuse existing destinations and execute once.
Observe the same PID and creation time after tool timeouts. Preserve execution
failures and application disagreements without dropping cases, changing labels
or weakening the existing numerical gate.

`close_report.py` closes and reports the retained dated campaign, including its
specific failed-attempt and preparation-reuse provenance. It is not a generic
closure command for a fresh reproduction. Its `close` and `report` actions both
take `--artifact`; `report` also accepts a new `--destination`. Report destinations
must not already contain the dated output files.

The first inference attempt stopped after nine native requests because a Windows
reader temporarily denied replacement of the status file. Its complete failure
and process termination are retained. The corrected writer retries that specific
sharing failure for at most one second; a real-lock test checks transient recovery
and bounded failure. Recognition cases and policies remain unchanged.
