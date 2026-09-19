# Labeled Parakeet and Whisper application check

This opt-in lane evaluates the public `ParakeetTranscriber` and
`WhisperTranscriber` APIs on a fixed subset of English read speech. It checks
recognition errors against human transcripts separately from agreement with
native ONNX Runtime. It does not replace either model's complete numerical
qualification, and is not the official full LibriSpeech benchmark.

[dataset.json](dataset.json) pins the CC-BY-4.0
[LibriSpeech clean test data](https://huggingface.co/datasets/openslr/librispeech_asr/tree/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1/clean/test).
Selection uses speaker ID, recording ID and duration, before recognition:
take the first ten numeric speaker IDs having both a 4–10 second and a
12–25 second recording, then the lexicographically first recording in each
band. Speaker 1272 is excluded because the earlier demonstration fixture
uses that speaker. This yields twenty recordings; related training-data
exposure is not established by this test-split label.

Use a separate Python environment, `ffmpeg` on PATH, .NET 10 and the pinned
local [Parakeet](../../parakeet/transcribe/README.md) and
[Whisper](../../whisper/README.md) assets. Download only the dataset files:

```powershell
python -m pip install -r tests/audio/accuracy/requirements.txt
hf download openslr/librispeech_asr README.md clean/test/0000.parquet --type dataset --revision 71cacbfb7e2354c4226d01e70d77d5fca3d04ba1 --local-dir models/librispeech-accuracy
python tests/audio/accuracy/prepare.py --parquet models/librispeech-accuracy/clean/test/0000.parquet --output artifacts/asr-accuracy/inputs
python -m unittest discover -s tests/audio/accuracy -p test_score.py -v
```

Preparation verifies the pinned parquet, inventories every candidate and
freezes selection before generating features. It preserves the selected FLAC
files and human transcripts, requires identical complete float PCM from
SoundFile and ffmpeg, and creates the pinned NumPy Whisper features. Model
inference is not involved in selection. The frontend is imported before
PyArrow to avoid a native-library import-order failure observed on Windows
when optional Torch is already installed; feature extraction explicitly uses
the NumPy path.

Generate native results separately. Parakeet additionally needs the pinned
onnx-asr source checkout described in its existing reference instructions;
only its verified greedy-loop and text-cleanup methods are executed. The new
generator preserves the existing seven-case full-tensor lane unchanged.

```powershell
python tests/audio/accuracy/parakeet_reference.py --models models/parakeet-tdt-0.6b-v3 --audio artifacts/asr-accuracy/inputs/audio.json --reference-source external/onnx-asr --output artifacts/asr-accuracy/native-parakeet
python tests/whisper/generate_transcription_reference.py --models models/whisper-large-v3-turbo --audio artifacts/asr-accuracy/inputs/audio.json --output artifacts/asr-accuracy/native-whisper
dotnet build tests/audio/accuracy/AccuracyReplay.csproj -c Release --tl:off --nologo -v minimal
dotnet tests/audio/accuracy/bin/Release/net10.0/AccuracyReplay.dll parakeet models/parakeet-tdt-0.6b-v3 artifacts/asr-accuracy/inputs/audio.json artifacts/asr-accuracy/native-parakeet/manifest.json artifacts/asr-accuracy/managed-parakeet
dotnet tests/audio/accuracy/bin/Release/net10.0/AccuracyReplay.dll whisper models/whisper-large-v3-turbo artifacts/asr-accuracy/inputs/audio.json artifacts/asr-accuracy/native-whisper/manifest.json artifacts/asr-accuracy/managed-whisper
```

Run only one model process at a time. Each managed process checks pinned model
and input identities, executes twenty public PCM requests and repeats the
first request. It retains the actual earlier inputs and returned results,
checks them after subsequent requests, and rejects a loaded native ORT module.
Parakeet compares text, tokens, frame positions, duration predictions, stop
reason and decoder counts. Whisper compares text, tokens, stop reason and the
no-speech decision, retaining its probability values. Every case is saved,
including mismatches. Exit code 2 means application disagreement; preserve
those results and score them rather than dropping recordings.

```powershell
python tests/audio/accuracy/score.py --audio artifacts/asr-accuracy/inputs/audio.json --parakeet-native artifacts/asr-accuracy/native-parakeet/manifest.json --parakeet-managed artifacts/asr-accuracy/managed-parakeet/result.json --whisper-native artifacts/asr-accuracy/native-whisper/manifest.json --whisper-managed artifacts/asr-accuracy/managed-whisper/result.json --output artifacts/asr-accuracy/score.json
```

The scorer independently checks selection, transcript/recording hashes,
complete ordered coverage and application comparisons. Text normalization is
fixed for both models: Unicode NFKC, uppercase, preserve letters/numbers and
internal ASCII apostrophes, replace other characters with spaces, then
collapse whitespace. Numbers, abbreviations and proper names receive no
special correction. Character error includes normalized spaces. Word and
character error rates divide total edit errors by total reference length;
utterance percentages are not averaged. An independent dynamic-programming
distance checks JiWER's total errors. JiWER supplies one optimal alignment's
substitution/deletion/insertion counts; ambiguous alignments can have different
counts with the same total distance.

All output directories and the final score must be new. Keep failed or partial
attempts. These are development-only tools; native ORT and Python packages do
not ship in the managed library. This small, clean English subset does not
qualify noisy, conversational, multilingual or long-recording recognition.
