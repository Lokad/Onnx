namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;

/// <summary>Explicit greedy transcription policy; timestamps and translation are not generated.</summary>
public sealed record WhisperTranscriptionOptions(string Language, int MaxNewTokens,
    double? NoSpeechThreshold, double? LogProbabilityThreshold)
{
    /// <summary>Up to 444 generated tokens with the standard no-speech confidence policy.</summary>
    public static WhisperTranscriptionOptions ForLanguage(string language) => new WhisperTranscriptionOptions(language, 444, 0.6, -1.0);
}

/// <summary>The reason greedy decoding stopped.</summary>
public enum WhisperStopReason { EndToken, TokenLimit, SilentInput }

/// <summary>Text and immutable token IDs from one independent transcription request.</summary>
/// <remarks>Token IDs include EOS when generated. A no-speech decision empties Text,
/// but retains tokens and probabilities so callers can inspect that decision.
/// Empty or exactly zero PCM bypasses inference; its model probabilities are null.</remarks>
public sealed record WhisperTranscription(string Text, IReadOnlyList<int> TokenIds,
    WhisperStopReason StopReason, bool SkippedAsNoSpeech, double? NoSpeechProbability, double? AverageLogProbability);

internal sealed class WhisperGeneration
{
    internal const int VocabularySize = 51866;
    internal const int MaximumLength = 448;
    readonly Dictionary<string, int> languages = new Dictionary<string, int>(StringComparer.Ordinal);
    readonly HashSet<int> suppressed;
    readonly HashSet<int> beginSuppressed;
    readonly WhisperTokenizer tokenizer;
    readonly int start, end, transcribe, noTimestamps, noSpeech;

    internal WhisperGeneration(Stream configuration, WhisperTokenizer tokenizer)
    {
        this.tokenizer = tokenizer;
        using var document = JsonDocument.Parse(configuration);
        var config = document.RootElement;
        start = config.GetProperty("decoder_start_token_id").GetInt32();
        end = config.GetProperty("eos_token_id").GetInt32();
        transcribe = config.GetProperty("task_to_id").GetProperty("transcribe").GetInt32();
        noTimestamps = config.GetProperty("no_timestamps_token_id").GetInt32();
        noSpeech = tokenizer.SpecialId("<|nospeech|>");
        if (start != tokenizer.SpecialId("<|startoftranscript|>") || end != tokenizer.SpecialId("<|endoftext|>")
            || transcribe != tokenizer.SpecialId("<|transcribe|>") || noTimestamps != tokenizer.SpecialId("<|notimestamps|>")
            || config.GetProperty("max_length").GetInt32() != MaximumLength)
            throw new InvalidDataException("Whisper generation configuration and tokenizer disagree.");
        foreach (var entry in config.GetProperty("lang_to_id").EnumerateObject())
        {
            if (!entry.Name.StartsWith("<|", StringComparison.Ordinal) || !entry.Name.EndsWith("|>", StringComparison.Ordinal)
                || entry.Value.GetInt32() != tokenizer.SpecialId(entry.Name))
                throw new InvalidDataException("Invalid Whisper language mapping.");
            languages.Add(entry.Name.Substring(2, entry.Name.Length - 4), entry.Value.GetInt32());
        }
        suppressed = ReadTokens(config.GetProperty("suppress_tokens"));
        beginSuppressed = ReadTokens(config.GetProperty("begin_suppress_tokens"));
        foreach (int id in new[] { start, end, transcribe, noTimestamps, noSpeech }.Concat(languages.Values))
            if (id < 0 || id >= VocabularySize) throw new InvalidDataException("Special token is outside the vocabulary.");
    }

    static HashSet<int> ReadTokens(JsonElement value)
    {
        var result = new HashSet<int>();
        foreach (var token in value.EnumerateArray())
        {
            int id = token.GetInt32();
            if (id < 0 || id >= VocabularySize) throw new InvalidDataException("Suppressed token is outside the vocabulary.");
            result.Add(id);
        }
        return result;
    }

    internal void Validate(WhisperTranscriptionOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (options.Language is null || !languages.ContainsKey(options.Language))
            throw new ArgumentException("Use a language code from this model's generation configuration.", nameof(options));
        if (options.MaxNewTokens < 1 || options.MaxNewTokens > MaximumLength - 4)
            throw new ArgumentOutOfRangeException(nameof(options), "Generated token limit must be between 1 and 444.");
        if (options.NoSpeechThreshold is double ns && (!double.IsFinite(ns) || ns < 0 || ns > 1)
            || options.LogProbabilityThreshold is double lp && (!double.IsFinite(lp) || lp > 0))
            throw new ArgumentOutOfRangeException(nameof(options), "Invalid probability threshold.");
    }

    internal WhisperTranscription Decode(Tensor<float> hidden, WhisperTranscriptionOptions options,
        Func<bool, Dictionary<string, ITensor>, IReadOnlyDictionary<string, ITensor>> execute, CancellationToken cancellation)
    {
        Validate(options);
        var tokens = new List<int>();
        var cross = new Dictionary<string, ITensor>();
        var self = new Dictionary<string, ITensor>();
        double noSpeechProbability = 0, sumLogProbability = 0;
        var stop = WhisperStopReason.TokenLimit;
        for (int step = 0; step < options.MaxNewTokens; step++)
        {
            cancellation.ThrowIfCancellationRequested();
            bool first = step == 0;
            long[] ids = first ? new long[] { start, languages[options.Language], transcribe, noTimestamps } : new long[] { tokens[tokens.Count - 1] };
            var feeds = new Dictionary<string, ITensor> { ["input_ids"] = new DenseTensor<long>(ids, new[] { 1, ids.Length }) };
            if (first) feeds.Add("encoder_hidden_states", hidden);
            else
            {
                foreach (var entry in cross) feeds.Add(entry.Key, entry.Value);
                foreach (var entry in self) feeds.Add(entry.Key, entry.Value);
            }
            var outputs = execute(first, feeds);
            var logits = RequireFloat(outputs, "logits", new[] { 1, ids.Length, VocabularySize }).ToArray();
            for (int i = 0; i < logits.Length; i++)
                if (!float.IsFinite(logits[i])) throw new InvalidDataException("Whisper returned nonfinite logits.");
            if (first) noSpeechProbability = Math.Exp(LogProbability(logits.AsSpan(0, VocabularySize), noSpeech));
            var last = logits.AsSpan(logits.Length - VocabularySize);
            foreach (int id in suppressed) last[id] = float.NegativeInfinity;
            if (first) foreach (int id in beginSuppressed) last[id] = float.NegativeInfinity;
            // A plain text request cannot emit language/task markers or timestamps.
            for (int id = end + 1; id < last.Length; id++) last[id] = float.NegativeInfinity;
            int chosen = 0;
            for (int id = 1; id < last.Length; id++) if (last[id] > last[chosen]) chosen = id;
            if (!float.IsFinite(last[chosen])) throw new InvalidDataException("Whisper suppression removed every token.");
            sumLogProbability += LogProbability(last, chosen);
            tokens.Add(chosen);
            if (chosen == end) { stop = WhisperStopReason.EndToken; break; }
            if (step + 1 == options.MaxNewTokens) break;
            var nextSelf = new Dictionary<string, ITensor>();
            for (int layer = 0; layer < 4; layer++)
                foreach (string kind in new[] { "key", "value" })
                {
                    string decoder = layer + ".decoder." + kind;
                    nextSelf.Add("past_key_values." + decoder, RequireFloat(outputs, "present." + decoder, new[] { 1, 20, 4 + step, 64 }));
                    if (first)
                    {
                        string encoder = layer + ".encoder." + kind;
                        cross.Add("past_key_values." + encoder, RequireFloat(outputs, "present." + encoder, new[] { 1, 20, hidden.Dimensions[1], 64 }));
                    }
                }
            self = nextSelf;
        }
        // The reference includes EOS in the denominator even on length termination.
        int textCount = tokens.Count - (stop == WhisperStopReason.EndToken ? 1 : 0);
        double average = sumLogProbability / (textCount + 1);
        bool skipped = options.NoSpeechThreshold is double threshold && noSpeechProbability > threshold
            && !(options.LogProbabilityThreshold is double logThreshold && average > logThreshold);
        string text = tokenizer.Decode(tokens, end);
        return new WhisperTranscription(skipped ? string.Empty : text, Array.AsReadOnly(tokens.ToArray()), stop, skipped, noSpeechProbability, average);
    }

    static double LogProbability(ReadOnlySpan<float> logits, int chosen)
    {
        float maximum = float.NegativeInfinity;
        foreach (float value in logits) maximum = Math.Max(maximum, value);
        double total = 0;
        foreach (float value in logits) total += Math.Exp((double)value - maximum);
        return (double)logits[chosen] - maximum - Math.Log(total);
    }

    internal static Tensor<float> RequireFloat(IReadOnlyDictionary<string, ITensor> outputs, string name, int[] shape)
    {
        if (!outputs.TryGetValue(name, out var value) || value is not Tensor<float> tensor || !tensor.Dimensions.SequenceEqual(shape))
            throw new InvalidDataException("Unexpected Whisper output contract: " + name);
        return tensor;
    }
}
