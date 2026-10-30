# Lokad.Onnx.Data

Text, image and audio helpers built from source; this project is nonpackable and
is separate from the core NuGet package. Model file loading uses the core's
integrated `OnnxImport` API. File-explicit entry points use local
assets. EnsureMe5sTokenizer and the named BERT loader explicitly acquire missing
tokenizers; subsequent calls reuse their caches. The me5s lookup checks the binary
cache, then models/multilingual-e5-small/sentencepiece.bpe.model by ancestor search.

## EmbeddingGemma 2

`EmbeddingGemma2Embedder` loads the local prepared FP32 bundle once and returns
owned embedding arrays. Query and document methods apply different retrieval
prompts; use the matching method when producing each kind of vector.

```csharp
using Lokad.Onnx;

var embedder = new EmbeddingGemma2Embedder("models/embeddinggemma-2/prepared-fp32");
var query = embedder.EmbedQuery("Which planet is known as the Red Planet?", 256,
    CancellationToken.None);
var documents = embedder.EmbedDocuments(
    new[] { new EmbeddingGemma2Document("Mars is known as the Red Planet.", "Mars") },
    256, CancellationToken.None);
```

Supported dimensions are 128, 256, 512 and 768. Defaults use 768 dimensions and
a 512-token bound including prompt and BOS/EOS; overflow is rejected. Batches
are bounded, calls serialize, and cancellation is checked between batches.
See the [embedding guide](../../tests/embeddinggemma2/README.md) for prepared
assets, tokenizer setup, space identity and complete Python/native qualification.

## Text (Text.cs)

- BERT (BertTokenize) and XLM-RoBERTa (RobertaTokenize*, LoadRobertaTokenizerFromFile) tokenizers producing input_ids / attention_mask (/ token_type_ids) tensors.
- RobertaTokenizeFromFile(text[s], tokenizerModelPath) is the file-explicit entry point; the shared instance cache is path-keyed and locked.
- CLI text args use --text <props> (e.g. --text me5s), resolved through GetTextTensors*; plain .txt files go through GetTextTensorsFromFileArg.

## Images (Images.cs)

- File args use the path::format convention via Data.GetInputTensorsFromFileArgs (also the CLI run input path). Supported formats:
  - mnist: grayscale, 28x28, single channel, values in [0, 1].
  - dinov2: 224x224 RGB, channels-first, values in [0, 1], no mean/std normalization.
  - dinov3: 224x224 RGB, channels-first, rescale to [0, 1] then ImageNet mean (0.485, 0.456, 0.406) / std (0.229, 0.224, 0.225) normalization, matching models/dinov3-vits16/preprocessor_config.json. Resize uses the Triangle sampler, which approximates the reference bilinear resample (ImageSharp 3 removed Bilinear).
  - WxH (e.g. file.png::224:224): stretch-resize to the given dimensions, single channel like mnist.
- With --save-input the (possibly resized) input image is written next to the source file for inspection.

## Audio and Whisper

`WaveAudio.ReadMono(stream, maximumDuration)` reads a single little-endian
RIFF/WAVE form from a readable, seekable stream and leaves the stream open.
It returns owned float samples and their sample rate. Supported recordings have
one or two channels, rates 8000..192000 Hz, PCM8/16/24/32 or IEEE float32/64,
including their extensible subtypes. Two channels are averaged. Float volume is
preserved without clipping; nonfinite or unrepresentable values are rejected.
Compressed WAV, RF64 and big-endian RIFX are unsupported. The explicit duration
limit is enforced before allocating the sample array; excess audio is rejected.

`AudioResampler.Resample(samples, sourceRate, destinationRate)` returns owned
mono PCM, using a centered low-pass FIR filter to suppress aliasing. Rates are
8000..192000 Hz. Output length is `ceil(length * destinationRate / sourceRate)`;
the input time origin is preserved and samples outside the clip are zero.
Same-rate calls copy exactly. There is no volume normalization or clipping.
The filter uses reduced integer factors, Kaiser beta 8.6, half length
`32 * max(up, down)` and cutoff `0.94 / max(up, down)`, normalized at DC.

```csharp
using var input = File.OpenRead("recording.wav");
var audio = WaveAudio.ReadMono(input, TimeSpan.FromSeconds(30));
var pcm = AudioResampler.Resample(audio.Samples, audio.SampleRate, 16000);
var model = new WhisperTranscriber("models/whisper-large-v3-turbo");
var result = model.Transcribe(pcm, 16000,
    WhisperTranscriptionOptions.ForLanguage("fr"), CancellationToken.None);
Console.WriteLine(result.Text);
```

Whisper uses local FP32 split assets, explicit language and greedy decoding.
The short `Transcribe` API rejects more than 30 seconds and retains
token/stop/no-speech metadata. `TranscribeRecording` accepts bounded recordings
through ten minutes and returns segment timestamps; see the
[recording policy](../../tests/whisper/recording/README.md).
See [component and application qualification](../../tests/whisper/README.md)
for the pinned export and the unresolved full encoder/logit numerical gate.

`ParakeetTranscriber` provides short and bounded-recording transcription with
automatic language detection. See its [short API](../../tests/parakeet/transcribe/README.md)
and [recording policy](../../tests/parakeet/recording/README.md) for model paths,
quiet/hard window boundaries, partial results and qualification limits.

## Speaker diarization

`Community1Diarizer` accepts normalized mono 16 kHz PCM through ten minutes and
local Community-1 segmentation, split WeSpeaker, projection and PLDA paths. It
returns owned ordinary and exclusive speaker intervals with matching centroids.
See [the API/CLI contract and qualification](../../tests/pyannote/diarization/README.md)
for local asset preparation, deterministic vote ties, recording-bound intervals,
explicit no-data results and the retained numerical failures.

