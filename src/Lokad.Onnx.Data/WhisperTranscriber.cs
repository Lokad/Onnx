namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;

/// <summary>Managed greedy transcription using the FP32 Whisper Large V3 Turbo split export.</summary>
/// <remarks>Loads local encoder_model.onnx, decoder_model.onnx and decoder_with_past_model.onnx
/// under the model directory's onnx subdirectory. Requests on one instance serialize to bound
/// memory. Execution contexts are reused; each request creates independent attention-cache state.
/// Encoder numerical qualification is documented separately in tests/whisper/README.md.</remarks>
public sealed class WhisperTranscriber
{
    /// <summary>Maximum duration accepted by TranscribeRecording: ten minutes at 16 kHz.</summary>
    public const int MaximumRecordingSamples = WhisperAudio.SampleRate * 600;
    readonly ComputationalGraph encoder, firstDecoder, pastDecoder;
    readonly WhisperGeneration generation;
    internal long SharedDecoderWeightBytes { get; }
    readonly GraphExecution encodingExecution, firstExecution, pastExecution;
    readonly object gate = new object();

    /// <summary>Loads the local models and metadata; never downloads or loads native ORT.</summary>
    public WhisperTranscriber(string modelDirectory)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelDirectory);
        using (var config = JsonDocument.Parse(File.ReadAllText(Path.Combine(modelDirectory, "config.json"))))
        {
            var root = config.RootElement;
            foreach (var expected in new Dictionary<string, int> { ["d_model"] = 1280, ["encoder_layers"] = 32,
                ["decoder_layers"] = 4, ["decoder_attention_heads"] = 20, ["max_target_positions"] = 448,
                ["num_mel_bins"] = 128, ["vocab_size"] = WhisperGeneration.VocabularySize })
                if (root.GetProperty(expected.Key).GetInt32() != expected.Value)
                    throw new NotSupportedException("This API requires the Whisper Large V3 Turbo split FP32 configuration.");
        }
        using var tokenizerFile = File.OpenRead(Path.Combine(modelDirectory, "tokenizer.json"));
        var tokenizer = new WhisperTokenizer(tokenizerFile);
        using var generationFile = File.OpenRead(Path.Combine(modelDirectory, "generation_config.json"));
        generation = new WhisperGeneration(generationFile, tokenizer);
        // Three split graphs already retain their original and folded weights. Bound
        // optional packed clones to leave memory for repeated requests on a 16 GB host.
        encoder = Load(modelDirectory, "encoder_model.onnx", 256L * 1024 * 1024);
        firstDecoder = Load(modelDirectory, "decoder_model.onnx", 64L * 1024 * 1024);
        pastDecoder = Load(modelDirectory, "decoder_with_past_model.onnx", 64L * 1024 * 1024);
        RequireInputs(encoder, new[] { "input_features" });
        RequireInputs(firstDecoder, new[] { "input_ids", "encoder_hidden_states" });
        var pastNames = new List<string> { "input_ids" };
        for (int layer = 0; layer < 4; layer++)
            foreach (string attention in new[] { "decoder", "encoder" })
                foreach (string kind in new[] { "key", "value" }) pastNames.Add($"past_key_values.{layer}.{attention}.{kind}");
        RequireInputs(pastDecoder, pastNames);
        SharedDecoderWeightBytes = WhisperDecoderWeights.Share(firstDecoder, pastDecoder);
        encodingExecution = encoder.CreateExecution(ExecutionOptions.Memory, 512L * 1024 * 1024);
        firstExecution = firstDecoder.CreateExecution(ExecutionOptions.Memory, 128L * 1024 * 1024);
        pastExecution = pastDecoder.CreateExecution(ExecutionOptions.Memory, 128L * 1024 * 1024);
    }

    static ComputationalGraph Load(string directory, string name, long packedWeightBytes) => OnnxImport.Load(Path.Combine(directory, "onnx", name), packedWeightBytes)
        ?? throw new InvalidDataException("Could not load " + name + ": " + OnnxImport.LastErrorMessage, OnnxImport.LastErrorCause);

    static void RequireInputs(ComputationalGraph graph, IEnumerable<string> names)
    {
        if (!graph.Inputs.Keys.OrderBy(x => x, StringComparer.Ordinal).SequenceEqual(names.OrderBy(x => x, StringComparer.Ordinal)))
            throw new NotSupportedException("The ONNX graph is not the supported Whisper split export.");
    }

    /// <summary>Transcribes at most 30 seconds of finite mono 16 kHz PCM without changing the input.</summary>
    /// <remarks>Longer audio is rejected. Cancellation is checked between model calls; an in-flight
    /// model call completes first. Empty input is valid silence. No resampling or automatic language
    /// detection occurs. The result distinguishes EOS, token limits and the no-speech decision.</remarks>
    public WhisperTranscription Transcribe(ReadOnlySpan<float> samples, int sampleRate,
        WhisperTranscriptionOptions options, CancellationToken cancellation)
    {
        generation.Validate(options);
        if (samples.Length > WhisperAudio.SampleCount)
            throw new ArgumentOutOfRangeException(nameof(samples), "Transcribe accepts at most 30 seconds; split longer recordings explicitly.");
        if (sampleRate != WhisperAudio.SampleRate)
            throw new ArgumentOutOfRangeException(nameof(sampleRate), "Transcribe requires mono 16000 Hz PCM.");
        cancellation.ThrowIfCancellationRequested();
        bool silent = true;
        foreach (float sample in samples)
        {
            if (!float.IsFinite(sample)) throw new ArgumentException("Audio samples must be finite.", nameof(samples));
            silent &= sample == 0;
        }
        // The exported model can hallucinate text on exact digital silence.
        // This guard makes no energy threshold or learned VAD assumption.
        if (silent) return new WhisperTranscription(string.Empty, Array.AsReadOnly(Array.Empty<int>()), WhisperStopReason.SilentInput, true, null, null);
        lock (gate)
        {
            cancellation.ThrowIfCancellationRequested();
            return TranscribeWindow(samples, sampleRate, options, false, cancellation);
        }
    }

    /// <summary>Transcribes up to ten minutes of finite mono 16 kHz PCM using segment timestamps.</summary>
    /// <remarks>Windows advance to completed model timestamps, re-reading unfinished boundary audio.
    /// The result retains every window decision and distinguishes completed processing from limits.
    /// One instance serializes calls. Cancellation occurs between model calls, with no partial result.
    /// Segment timing is estimated; numerical and broader long-audio qualification are documented separately.</remarks>
    public WhisperRecording TranscribeRecording(ReadOnlySpan<float> samples, int sampleRate,
        WhisperRecordingOptions options, CancellationToken cancellation)
    {
        generation.ValidateRecording(options);
        if (sampleRate != WhisperAudio.SampleRate) throw new ArgumentOutOfRangeException(nameof(sampleRate), "Recording requires mono 16000 Hz PCM.");
        if (samples.Length > MaximumRecordingSamples) throw new ArgumentOutOfRangeException(nameof(samples), "Recording accepts at most ten minutes.");
        cancellation.ThrowIfCancellationRequested();
        foreach (float sample in samples)
            if (!float.IsFinite(sample)) throw new ArgumentException("Audio samples must be finite.", nameof(samples));
        lock (gate)
        {
            cancellation.ThrowIfCancellationRequested();
            float[] owned = samples.ToArray();
            return WhisperRecordingPolicy.Run(owned.Length, options.MaxWindows, (start, count) =>
            {
                ReadOnlySpan<float> window = owned.AsSpan(start, count);
                bool silent = true;
                foreach (float sample in window) silent &= sample == 0;
                if (silent) return new WhisperTranscription(string.Empty, Array.AsReadOnly(Array.Empty<int>()), WhisperStopReason.SilentInput, true, null, null);
                return TranscribeWindow(window, sampleRate, options.Decoding, true, cancellation);
            }, generation.DecodeText, cancellation);
        }
    }

    WhisperTranscription TranscribeWindow(ReadOnlySpan<float> samples, int sampleRate,
        WhisperTranscriptionOptions options, bool timestamps, CancellationToken cancellation)
    {
        var features = WhisperAudio.LogMelSpectrogram(samples, sampleRate);
        var encoding = encodingExecution;
        var first = firstExecution;
        var past = pastExecution;
        try
        {
            var outputs = Execute(encoding, new Dictionary<string, ITensor> { ["input_features"] = features });
            var hidden = WhisperGeneration.RequireFloat(outputs, "last_hidden_state", new[] { 1, 1500, 1280 });
            cancellation.ThrowIfCancellationRequested();
            return timestamps
                ? generation.DecodeTimestamps(hidden, options, (initial, feeds) => Execute(initial ? first : past, feeds), cancellation)
                : generation.Decode(hidden, options, (initial, feeds) => Execute(initial ? first : past, feeds), cancellation);
        }
        finally { encoding.Reset(); first.Reset(); past.Reset(); }
    }

    static IReadOnlyDictionary<string, ITensor> Execute(GraphExecution context, Dictionary<string, ITensor> feeds)
    {
        context.Reset();
        if (!context.Execute(feeds, true, ExecutionProvider.CPU, ExecutionOptions.Memory))
            throw new InvalidDataException(context.LastErrorMessage, context.LastErrorCause);
        return context.Outputs.ToDictionary(p => p.Key, p => p.Value
            ?? throw new InvalidDataException("Missing Whisper output: " + p.Key));
    }
}
