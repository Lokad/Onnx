namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;

/// <summary>Managed greedy transcription with the local FP32 Parakeet TDT 0.6B V3 export.</summary>
/// <remarks>Requires nemo128.onnx, encoder-model.onnx and its external weights,
/// decoder_joint-model.onnx, config.json and vocab.txt. One instance serializes requests;
/// each request owns independent execution contexts and recurrent states.</remarks>
public sealed class ParakeetTranscriber
{
    public const int SampleRate = 16000;
    public const int MaximumSamples = SampleRate * 30;
    readonly ComputationalGraph frontend, encoder, decoder;
    readonly ParakeetGeneration generation;
    readonly object gate = new object();

    /// <summary>Loads local model assets without downloading or loading native ONNX Runtime.</summary>
    public ParakeetTranscriber(string modelDirectory)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelDirectory);
        using (var configuration = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(modelDirectory, "config.json"))))
        {
            var value = configuration.RootElement;
            if (value.GetProperty("model_type").GetString() != "nemo-conformer-tdt"
                || value.GetProperty("features_size").GetInt32() != 128 || value.GetProperty("subsampling_factor").GetInt32() != 8
                || value.TryGetProperty("max_tokens_per_step", out var limit) && limit.GetInt32() != 10)
                throw new NotSupportedException("This API requires the Parakeet TDT 0.6B V3 FP32 split export.");
        }
        using (var vocabulary = File.OpenRead(Path.Combine(modelDirectory, "vocab.txt")))
            generation = new ParakeetGeneration(new ParakeetVocabulary(vocabulary));
        frontend = Load(modelDirectory, "nemo128.onnx", 0);
        encoder = Load(modelDirectory, "encoder-model.onnx", 256L * 1024 * 1024);
        decoder = Load(modelDirectory, "decoder_joint-model.onnx", 64L * 1024 * 1024);
        RequireNames(frontend, new[] { "waveforms", "waveforms_lens" }, new[] { "features", "features_lens" });
        RequireNames(encoder, new[] { "audio_signal", "length" }, new[] { "outputs", "encoded_lengths" });
        RequireNames(decoder, new[] { "encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2" },
            new[] { "outputs", "prednet_lengths", "output_states_1", "output_states_2" });
    }

    static ComputationalGraph Load(string directory, string name, long packedWeightBytes) =>
        OnnxImport.Load(Path.Combine(directory, name), packedWeightBytes)
        ?? throw new InvalidDataException("Could not load " + name + ": " + OnnxImport.LastErrorMessage, OnnxImport.LastErrorCause);

    static void RequireNames(ComputationalGraph graph, string[] inputs, string[] outputs)
    {
        if (!graph.Inputs.Keys.OrderBy(v => v, StringComparer.Ordinal).SequenceEqual(inputs.OrderBy(v => v, StringComparer.Ordinal))
            || !graph.Outputs.Keys.OrderBy(v => v, StringComparer.Ordinal).SequenceEqual(outputs.OrderBy(v => v, StringComparer.Ordinal)))
            throw new NotSupportedException("The graph is not the supported Parakeet split export.");
    }

    /// <summary>Transcribes finite mono 16 kHz PCM, from 257 samples through 30 seconds.</summary>
    /// <remarks>Empty or exact digital silence returns SilentInput without inference. Other input
    /// shorter than 257 samples is rejected. Language selection is intrinsic to this multilingual
    /// model. Cancellation is checked between graph calls; an in-flight call completes first.</remarks>
    public ParakeetTranscription Transcribe(ReadOnlySpan<float> samples, int sampleRate,
        ParakeetTranscriptionOptions options, CancellationToken cancellation)
    {
        ParakeetGeneration.Validate(options);
        if (sampleRate != SampleRate) throw new ArgumentOutOfRangeException(nameof(sampleRate), "Parakeet requires mono 16000 Hz PCM.");
        if (samples.Length > MaximumSamples) throw new ArgumentOutOfRangeException(nameof(samples), "Parakeet accepts at most 30 seconds per request.");
        cancellation.ThrowIfCancellationRequested();
        bool silent = true;
        foreach (float value in samples)
        {
            if (!float.IsFinite(value)) throw new ArgumentException("Audio samples must be finite.", nameof(samples));
            silent &= value == 0;
        }
        if (silent) return new ParakeetTranscription(string.Empty, Array.AsReadOnly(Array.Empty<int>()),
            Array.AsReadOnly(Array.Empty<int>()), Array.AsReadOnly(Array.Empty<int>()), ParakeetStopReason.SilentInput, 0, 0);
        if (samples.Length < 257) throw new ArgumentOutOfRangeException(nameof(samples), "Nonsilent Parakeet input must contain at least 257 samples.");
        lock (gate)
        {
            cancellation.ThrowIfCancellationRequested();
            var preprocessing = frontend.CreateExecution(ExecutionOptions.Memory);
            var encoding = encoder.CreateExecution(ExecutionOptions.Memory);
            var decoding = decoder.CreateExecution(ExecutionOptions.Memory);
            try
            {
                var prepared = Execute(preprocessing, new Dictionary<string, ITensor>
                {
                    ["waveforms"] = new DenseTensor<float>(samples.ToArray(), new[] { 1, samples.Length }),
                    ["waveforms_lens"] = new DenseTensor<long>(new[] { (long)samples.Length }, new[] { 1 })
                });
                int featureFrames = samples.Length / 160 + 1;
                var features = ParakeetGeneration.RequireFloat(prepared, "features", new[] { 1, 128, featureFrames });
                var featureLengths = RequireLength(prepared, "features_lens", featureFrames);
                cancellation.ThrowIfCancellationRequested();
                var encoded = Execute(encoding, new Dictionary<string, ITensor> { ["audio_signal"] = features, ["length"] = featureLengths });
                int frames = (featureFrames + 7) / 8;
                var hidden = ParakeetGeneration.RequireFloat(encoded, "outputs", new[] { 1, 1024, frames });
                RequireLength(encoded, "encoded_lengths", frames);
                return generation.Decode(hidden, frames, options, feeds => Execute(decoding, feeds), cancellation);
            }
            finally { preprocessing.Reset(); encoding.Reset(); decoding.Reset(); }
        }
    }

    static Tensor<long> RequireLength(IReadOnlyDictionary<string, ITensor> outputs, string name, long expected)
    {
        if (!outputs.TryGetValue(name, out var value) || value is not Tensor<long> tensor
            || !tensor.Dimensions.SequenceEqual(new[] { 1 }) || tensor.ToArray()[0] != expected)
            throw new InvalidDataException("Unexpected Parakeet length: " + name);
        return tensor;
    }

    static IReadOnlyDictionary<string, ITensor> Execute(GraphExecution context, Dictionary<string, ITensor> feeds)
    {
        context.Reset();
        if (!context.Execute(feeds, true, ExecutionProvider.CPU, ExecutionOptions.Memory))
            throw new InvalidDataException(context.LastErrorMessage, context.LastErrorCause);
        return context.Outputs.ToDictionary(p => p.Key, p => p.Value ?? throw new InvalidDataException("Missing Parakeet output: " + p.Key));
    }
}
