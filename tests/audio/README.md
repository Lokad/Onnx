# Recorded audio fixtures

The Whisper, Parakeet and pyannote lanes share three public recordings and one
synthetic silence input. [recordings.json](recordings.json) pins the original
repository revision, source WAV digests and all generated PCM/WAV/feature
digests. LibriSpeech, French and JFK recordings are drawn from
[Xenova/transformers.js-docs](https://huggingface.co/datasets/Xenova/transformers.js-docs/tree/fbe92bd97d48f3ec17779d8d8f2964e1c6bc7634).

Stage the files explicitly, then use a development Python environment with the
listed requirements and `ffmpeg` on PATH:

```powershell
hf download Xenova/transformers.js-docs librispeech_asr_demo_validation_0.wav french-audio.wav jfk.wav --repo-type dataset --revision fbe92bd97d48f3ec17779d8d8f2964e1c6bc7634 --local-dir models/audio-recordings
python -m pip install -r tests/audio/requirements.txt
python tests/audio/generate_recorded_inputs.py --recordings models/audio-recordings --output artifacts/audio-new
```

Use a new output directory. Generation never downloads files. It independently
decodes to mono 16 kHz float PCM and verifies the exact decoded hashes. It then
creates five WAV cases: English at 16 kHz, French stereo at 44.1 kHz, JFK stereo
PCM16 at 48 kHz, silence at 32 kHz, and JFK with a one-token transcription cap.
SciPy performs the explicit rational Kaiser resampling and independent decoding;
the pinned Whisper feature extractor supplies the reference mel features.

The resulting `audio.json` is accepted by `tests/whisper/generate_transcription_reference.py`
(`--audio`), `tests/parakeet/transcribe/generate_reference.py` (`--speech`),
`tests/parakeet/frontend/generate_reference.py` (`--speech-manifest`), and
`tests/pyannote/embedding/generate_reference.py` (`--speech`). New manifest
provenance changes its hash; generated WAV/PCM/feature bytes must reproduce the
pinned historical inputs exactly. The helper rejects changes instead of silently
substituting another waveform. These are small agreement fixtures, not a speech
accuracy benchmark.
