# Community-1 diarization

`Community1Diarizer` connects local FP32 segmentation, WeSpeaker embeddings and automatic VBx clustering into ordinary and exclusive speaker timelines. Ordinary intervals can overlap; exclusive intervals contain at most one active speaker. Product inference is managed and does not download assets or call native ORT, Torch or Python.

```csharp
var model = new Community1Diarizer(
    "models/pyannote-segmentation/segmentation/model.onnx",
    "models/pyannote-embedding/embedding_encoder.onnx",
    "models/pyannote-embedding/projection.onnx",
    "models/pyannote-community-1/plda.json");
Community1Diarization result = model.Diarize(pcm, 16000, cancellationToken);
foreach (var interval in result.Intervals)
    Console.WriteLine($"{interval.Start:F6}\t{interval.End:F6}\tSPEAKER_{interval.Speaker:D2}");
```

PCM must be finite, normalized to `[-1,1]`, mono 16 kHz, and at most ten minutes. Empty input returns `NoSpeech`. Other input, including digital silence, runs the segmentation model. Inputs are preserved; results and read-only centroid storage are owned. One instance serializes requests. Cancellation is checked between model calls and during managed postprocessing; an in-flight model call completes first. The duration limit bounds requests and does not promise a particular latency or maximum memory use for every recording.

`Completed` means the pipeline produced speaker intervals, not that human-labeled speaker accuracy is established. `NoUsableEmbeddings` distinguishes activity without sufficiently clean training vectors from `NoSpeech`. Forced speaker counts and their KMeans fallback are not supported.

## Local assets

Use the FP32 segmentation export from [FredrikKarlssonSpeech/pyannote-speaker-diarization-onnx](https://huggingface.co/FredrikKarlssonSpeech/pyannote-speaker-diarization-onnx/tree/0a7a3bf63c16c6718a7411a3e3fe01598fc46550), `segmentation/model.onnx`, SHA-256 `af62796adfc46ab36fb27c183e7fe6530a745c665b64e949acd73bf01a18a31a`. The model card identifies the Community-1 base model and CC-BY-4.0 license; retain its attribution and license with local assets.

The [speaker API instructions](../speaker/README.md) describe the pinned split encoder and local projection preparation. The [clustering instructions](../clustering/README.md) describe the Community-1 PLDA assets and preparation. Runtime reads the resulting ONNX and JSON files; Python packages are preparation/reference tools only. The selected pipeline configuration is Community-1 revision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`, with pyannote.audio 4.0.0 at `a1ed3bb0440d33d18622e1cb6b138431cfaf4f7a`.

## Time and speaker policy

Segmentation uses ten-second windows advancing one second, with a final zero-padded window when needed. Each window has 589 frames; their receptive-field duration is `991/16000` seconds and step is `270/16000` seconds. Interval boundaries use frame centers. Output intervals are clipped to the supplied recording duration and empty intervals are removed.

Powerset predictions use the first maximum class. Embedding masks exclude overlap when there are more than two clean segmentation frames, otherwise use the full speaker mask. This threshold follows the native model's 400-sample minimum. The pipeline uses native epsilon-regularized weighted statistics even for zero/single-frame masks. The standalone `WeSpeakerEmbedder.Extract` retains its separate two-positive-frame policy. Clustering training still requires at least 20 percent clean single-speaker frames.

Speaker counts average overlapping windows and round ties to even. Reconstruction sums cluster activity and selects the highest-vote speakers up to that count. Tied votes select the lowest canonical speaker label. This deterministic rule can differ from upstream NumPy's unstable default sort, whose ties vary with CPU dispatch. These differences must be recorded separately from untied disagreements.

As in the native reconstruction, overlap counts can exceed the number of clusters. Such extra speakers have zero centroids and `HasEmbedding=false`; they are not represented as learned speaker identities. Output speaker indices are contiguous, align with the returned centroids and are consistent across ordinary/exclusive intervals. Speaker numbers identify groups within one result, not named people or identities across recordings.

## CLI

```powershell
lonnx diarize recording.wav --segmentation models/pyannote-segmentation/segmentation/model.onnx --embedding models/pyannote-embedding/embedding_encoder.onnx --projection models/pyannote-embedding/projection.onnx --plda models/pyannote-community-1/plda.json --json
```

The CLI accepts the shared mono/stereo WAV formats at 8000..192000 Hz, mixes channels, resamples to 16 kHz and enforces ten minutes before inference. Floating volume is preserved; it is not silently clipped or normalized. JSON includes both timelines, speakers, duration, window count and status. Without `--json`, output is start/end seconds and `SPEAKER_nn` for ordinary intervals. Diagnostics go to stderr; cancellation exits 130.

## Qualification status

Initial local API and CLI replays match all five reference cases: English, French, JFK, a two-recording sequence spanning four windows, and silence. Final speaker centroids pass the scaled `1e-4` gate, maximum `1.90735e-6`. Independent timeline/window rules cover 97,198 values, with original unstable-sort outputs retained separately. Twenty ordinary API/rule tests and seven CLI tests cover boundaries, overlap, sparse masks, explicit no-data states, cancellation/recovery, concurrency, input/result ownership and invalid requests.

The complete initial numerical trace is **not qualified**. Among 67 arrays / 3,398,120 values, ten filterbank values exceed `1e-4` (maximum `2.53201e-4`) and 3,458 silence-segmentation scores fail (maximum `0.118817`). Binary masks, speaker counts, assignments, downstream embedding values and final timelines agree in these fixtures. These retained failures cannot be replaced by the functional timeline pass. Source-archive/AMD pipeline qualification, broader multi-speaker/long-recording fixtures and labeled diarization error measurement remain separate work.
