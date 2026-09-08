# Lokad.Onnx.Data

Text and image input helpers. File-explicit entry points are offline: they only read the paths handed to them and never download. Tokenizer acquisition is the explicit exception: EnsureMe5sTokenizer and the named BERT loader download the asset when absent, so call them up front; everything downstream reuses the locked cache offline. The me5s lookup resolves the binary cache first, then models/multilingual-e5-small/sentencepiece.bpe.model by ancestor search, so the documented setup needs no manual copy.

## Text (Text.cs)

- BERT (BertTokenize) and XLM-RoBERTa (RobertaTokenize*, LoadRobertaTokenizerFromFile) tokenizers producing input_ids / attention_mask (/ token_type_ids) tensors.
- RobertaTokenizeFromFile(text[s], tokenizerModelPath) is the file-explicit entry point; the shared instance cache is path-keyed and locked.
- CLI text args use --text <props> (e.g. --text me5s), resolved through GetTextTensors*; plain .txt files go through GetTextTensorsFromFileArg.

## Images (Images.cs)

- File args use the path::format convention via Data.GetInputTensorsFromFileArgs (also the CLI run input path). Supported formats:
  - mnist: grayscale, 28x28, single channel, values in [0, 1].
  - dinov2: 224x224 RGB, channels-first, values in [0, 1], no mean/std normalization.
  - dinov3: 224x224 RGB, channels-first, rescale to [0, 1] then ImageNet mean (0.485, 0.456, 0.406) / std (0.229, 0.224, 0.225) normalization, matching models/dinov3-vits16/preprocessor_config.json. Resize uses the Triangle sampler, which approximates the reference bilinear resample within 1 uint8 LSB (verified against a PIL reference; see PLAN.md).
  - WxH (e.g. file.png::224:224): stretch-resize to the given dimensions, single channel like mnist.
- With --save-input the (possibly resized) input image is written next to the source file for inspection.

