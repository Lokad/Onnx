# Natural meeting ASR evaluation

This lane prepares human word references for the same two independently selected
ten-minute AMI excerpts as the [diarization evaluation](../../pyannote/natural-meetings/README.md).
Recognition results are pending. It changes no model, production policy or
numerical tolerance.

Use the official [AMI manual annotations v1.6.2](https://groups.inf.ed.ac.uk/ami/download/),
CC BY 4.0, archive SHA256
`b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d`.
The original ISO-8859-1 word XML files preserve four speaker streams per meeting.
Only lexical words entirely inside each fixed crop enter the transcript;
punctuation and non-word events do not. Keep fillers, truncated words and original
spelling. Crossing-boundary words remain explicitly recorded. Input audio is
unchanged.

Sort words by start, end, speaker letter and original XML index. Apply the existing
five-language ASR normalization from `tests/audio/multilingual/common.py`.
This defines a chronological mixed-speaker WER observation. It is not an official
AMI ASR benchmark: overlapping speakers make a single word order ambiguous.
Retain all raw word times and speaker identities alongside the joined reference.
Aggregate only the two independent meeting excerpts; the thirty-second recovery
is correlated and excluded.

`prepare_labels.py --artifact <new-ASR-artifact> --diarization <existing-artifact>`
reads the downloaded archive and reuses the exact existing WAVs by hash. It
requires the lexical interval union from original XML to match the pinned
words-only RTTM references exactly. It extracts only ten named annotation files
and refuses existing outputs. Preparation performs no inference.

Run `python -B -m unittest discover -s tests/audio/natural-meetings` for the
encoding, metadata, ordering, boundary and interval-union tests.

The retained preparation has 1,113 normalized words for ES2004a and 1,348 for
IS1009a. Each excludes one lexical record crossing 600 seconds; thirteen and
twenty annotated truncated lexical records remain included, respectively.
The thirty-second recovery has 45 normalized words and excludes two crossing
records. `audit_labels.py --artifact <directory> --output <new-json>` independently
reconstructs every lexical record with a separate XML DOM parser and verifies
original archive bytes, audio hashes, complete ordering and normalized text.
The current preparation and three label-policy tests pass. These counts are reference
metadata, not recognition scores.

The recording runners preserve complete public results for Parakeet and Whisper.
Each model gets one fresh process and three sequential calls: ES2004a 600 seconds,
IS1009a 600 seconds, then ES2004a's first 30 seconds for recovery. Keep all input
arrays and returned objects alive and verify they remain unchanged after later calls.
Whisper uses explicit English and production recording defaults. No reference text
is supplied to either engine.

Native ORT runs on Windows CPU 2; managed inference runs on AMD CPU 2. These
accuracy replays do not provide a speed ratio between hosts. Native Parakeet also
checks the original upstream decoder trajectory. Whisper borrows the unchanged
qualified decode and timestamp function bodies, checks intermediate arrays for
finite float32 values, and retains all public decisions without large tensor dumps.
The saved recording validators require exact tokens, text, windows, timestamps,
seek and stop decisions. Whisper confidence differences are diagnostic under the
existing finite/probability/skip-policy contract. Existing full-tensor numerical
failures and the scaled `1e-4` gates remain separate.

Build with qualified product DLLs rather than rebuilding production:

```powershell
dotnet build tests/audio/natural-meetings/NaturalAsr.csproj -c Release --tl:off --nologo -v minimal -p:FrozenProductDirectory=<qualified-bin> -o <artifact>/bin
python -B tests/audio/natural-meetings/prepare_runtime.py --artifact <artifact>
python -B tests/audio/natural-meetings/prepare_auditors.py --artifact <artifact>
```

Preparation requires the pinned cached models, previous closed recording evidence,
and Python dependencies recorded in `prepare_runtime.py`. `prepare_auditors.py
--verify-existing` verifies a previously copied auditor set without overwriting it.
Before freezing, perform input-only supervised calls for both engines/families
(`supervise.py run --artifact <artifact> --engine <native|managed> --family
<parakeet|whisper> --mode inputs`) and retain their PCM/identity checks, the
offline Whisper wrapper replay, and the saved-validator replay.

`freeze.py --artifact <artifact>` requires committed source and a clean tree.
It binds the schedule, source, products, models, PCM, human references, scorers,
and preceding checks. Transfer only that exact payload to the new AMD artifact;
reuse the already deployed models and WAVs by hash. Run input-only checks there,
then launch `runtime/supervise.py launch --artifact <artifact> --engine <engine>`
once per host. The supervisor serializes Parakeet then Whisper and records process
births, CPU affinity and half-second resource samples. Required available memory
before launch is 20 GiB on Windows and 13 GiB on AMD; worker RSS limits are
20 GiB and 14 GiB respectively, with a two-hour limit per worker and at least
1 GiB available during execution. Observation timeouts never restart a worker.

Local verification includes four successful input-only workers, four offline
Whisper wrapper cases, fourteen retained validator cases and eight damaged-record
refusals. Two initial native input preflights failed before creating workers;
both remain recorded, and retries passed under the same memory requirements.
The runner builds with zero warnings/errors and the four label/resource test
methods pass. These checks perform no new neural inference.

After both campaigns finish, use the Python environment containing NumPy,
psutil 7, JiWER 4.0.0, RapidFuzz 3.14.6 and tokenizers. `collect.py --artifact
<artifact>` first verifies every owned AMD process birth is terminal, then
retrieves an exact, hash-checked inventory. It preserves failed runs and refuses
archive links, traversal paths, duplicate or missing entries. The two collection
test methods bring the lane's unit-test total to six.

Run `audit.py --artifact <artifact> --output <artifact>/audit.json`, then
`audit_resources.py --artifact <artifact> --output <artifact>/resource-audit.json`.
The first validates complete public objects and human word/character errors;
the second validates all four workers, samples, accounting and terminal identities.
`close.py --artifact <artifact>` independently rechecks recorded calls, scorers,
alignments, source identities and damaged-record refusals before writing a receipt.
`report.py --artifact <artifact>` renders only closed evidence. Every writer
refuses existing outputs. Application disagreement is recorded as a failure;
evidence closure does not turn it into a passing model qualification.

Attribution: Carletta et al., *The AMI meeting corpus: A pre-announcement* (2006).
Selection and diarization references use the pinned
[pyannote/BUT Speech@FIT setup](https://github.com/pyannote/AMI-diarization-setup/tree/67c2d539286e89f68952d5dcf83912bd9f01dfae).
