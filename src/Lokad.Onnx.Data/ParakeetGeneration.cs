namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;

/// <summary>Explicit limits for greedy Parakeet token-and-duration decoding.</summary>
public sealed record ParakeetTranscriptionOptions(int MaxTokens, int MaxTokensPerFrame)
{
    /// <summary>At most 4096 tokens and ten zero-duration emissions at one encoder frame.</summary>
    public static ParakeetTranscriptionOptions Default => new ParakeetTranscriptionOptions(4096, 10);
}

/// <summary>Why a Parakeet transcription request ended.</summary>
public enum ParakeetStopReason { EndOfAudio, TokenLimit, SilentInput }

/// <summary>Text and immutable token decisions for one independent request.</summary>
/// <remarks>Frame indices refer to the subsampled encoder, nominally 80 ms per frame.
/// Predicted duration values are 0..4 frames; these are model decisions, not word boundaries.
/// Blank predictions do not appear in the token lists. DecoderCalls includes blank predictions.</remarks>
public sealed record ParakeetTranscription(string Text, IReadOnlyList<int> TokenIds,
    IReadOnlyList<int> FrameIndices, IReadOnlyList<int> DurationFrames,
    ParakeetStopReason StopReason, int EncodedFrames, int DecoderCalls);

internal sealed class ParakeetVocabulary
{
    internal const int TokenCount = 8193;
    internal const int Blank = TokenCount - 1;
    readonly string[] pieces = new string[TokenCount];

    internal ParakeetVocabulary(Stream vocabulary)
    {
        ArgumentNullException.ThrowIfNull(vocabulary);
        using var reader = new StreamReader(vocabulary, new UTF8Encoding(false, true), false, 4096, true);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        int index = 0;
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            if (index == 0 && line.StartsWith('\uFEFF')) line = line.Substring(1);
            int separator = line.LastIndexOf(' ');
            if (index >= TokenCount || separator < 1
                || !int.TryParse(line.AsSpan(separator + 1), NumberStyles.None, CultureInfo.InvariantCulture, out int id)
                || id != index)
                throw new InvalidDataException("Parakeet vocabulary must contain consecutive IDs 0..8192.");
            string piece = line.Substring(0, separator);
            if (piece.Any(char.IsWhiteSpace) || !seen.Add(piece))
                throw new InvalidDataException("Parakeet vocabulary contains whitespace or a duplicate piece.");
            pieces[index++] = piece.Replace('\u2581', ' ');
        }
        if (index != TokenCount || pieces[Blank] != "<blk>" || pieces.Take(Blank).Contains("<blk>"))
            throw new InvalidDataException("Parakeet requires 8193 vocabulary entries with blank at 8192.");
    }

    internal string Decode(IEnumerable<int> tokens)
    {
        var text = new StringBuilder();
        foreach (int token in tokens)
        {
            if ((uint)token >= Blank) throw new InvalidDataException("Invalid emitted Parakeet token.");
            text.Append(pieces[token]);
        }
        // Match the reference's Python Unicode whitespace/word cleanup. A word
        // character is a Unicode letter/number or underscore, not a combining
        // mark. Enumerating scalars also handles letters outside the BMP.
        var scalars = text.ToString().EnumerateRunes().ToArray();
        text.Clear();
        for (int i = 0; i < scalars.Length; i++)
        {
            Rune value = scalars[i];
            if (!Rune.IsWhiteSpace(value)) text.Append(value.ToString());
            else if (i > 0 && i + 1 < scalars.Length && IsWord(scalars[i + 1])) text.Append(' ');
        }
        return text.ToString();
    }

    static bool IsWord(Rune value) => value.Value == '_' || Rune.GetUnicodeCategory(value) is
        UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter or UnicodeCategory.TitlecaseLetter
        or UnicodeCategory.ModifierLetter or UnicodeCategory.OtherLetter or UnicodeCategory.DecimalDigitNumber
        or UnicodeCategory.LetterNumber or UnicodeCategory.OtherNumber;
}

internal sealed class ParakeetGeneration
{
    readonly ParakeetVocabulary vocabulary;
    internal ParakeetGeneration(ParakeetVocabulary vocabulary) => this.vocabulary = vocabulary;

    internal static void Validate(ParakeetTranscriptionOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (options.MaxTokens < 1 || options.MaxTokens > 4096
            || options.MaxTokensPerFrame < 1 || options.MaxTokensPerFrame > 10)
            throw new ArgumentOutOfRangeException(nameof(options), "Parakeet limits are 1..4096 tokens and 1..10 tokens per frame.");
    }

    internal ParakeetTranscription Decode(Tensor<float> hidden, int frames, ParakeetTranscriptionOptions options,
        Func<Dictionary<string, ITensor>, IReadOnlyDictionary<string, ITensor>> execute, CancellationToken cancellation)
    {
        Validate(options);
        ArgumentNullException.ThrowIfNull(hidden);
        ArgumentNullException.ThrowIfNull(execute);
        if (hidden.Dimensions.Length != 3 || hidden.Dimensions[0] != 1 || hidden.Dimensions[1] != 1024
            || frames < 0 || frames > hidden.Dimensions[2])
            throw new InvalidDataException("Unexpected Parakeet encoder dimensions or valid length.");
        cancellation.ThrowIfCancellationRequested();
        float[] encoding = hidden.ToArray();
        if (encoding.Any(v => !float.IsFinite(v))) throw new InvalidDataException("Parakeet returned nonfinite encoder values.");
        int stride = hidden.Dimensions[2];
        Tensor<float> state1 = new DenseTensor<float>(new[] { 2, 1, 640 });
        Tensor<float> state2 = new DenseTensor<float>(new[] { 2, 1, 640 });
        var tokens = new List<int>();
        var positions = new List<int>();
        var durations = new List<int>();
        int frame = 0, emitted = 0, calls = 0;
        var stop = ParakeetStopReason.EndOfAudio;
        while (frame < frames)
        {
            cancellation.ThrowIfCancellationRequested();
            if (tokens.Count == options.MaxTokens) { stop = ParakeetStopReason.TokenLimit; break; }
            var current = new float[1024];
            for (int channel = 0; channel < current.Length; channel++) current[channel] = encoding[channel * stride + frame];
            var feeds = new Dictionary<string, ITensor>
            {
                ["encoder_outputs"] = new DenseTensor<float>(current, new[] { 1, 1024, 1 }),
                ["targets"] = new DenseTensor<int>(new[] { tokens.Count == 0 ? ParakeetVocabulary.Blank : tokens[tokens.Count - 1] }, new[] { 1, 1 }),
                ["target_length"] = new DenseTensor<int>(new[] { 1 }, new[] { 1 }),
                ["input_states_1"] = state1, ["input_states_2"] = state2
            };
            var outputs = execute(feeds);
            calls++;
            cancellation.ThrowIfCancellationRequested();
            var logits = RequireFloat(outputs, "outputs", new[] { 1, 1, 1, ParakeetVocabulary.TokenCount + 5 }).ToArray();
            var next1 = RequireFloat(outputs, "output_states_1", new[] { 2, 1, 640 });
            var next2 = RequireFloat(outputs, "output_states_2", new[] { 2, 1, 640 });
            if (!outputs.TryGetValue("prednet_lengths", out var length) || length is not Tensor<int> lengths
                || !lengths.Dimensions.SequenceEqual(new[] { 1 }) || lengths.ToArray()[0] != 1)
                throw new InvalidDataException("Unexpected Parakeet decoder length.");
            int token = ArgMax(logits.AsSpan(0, ParakeetVocabulary.TokenCount));
            int duration = ArgMax(logits.AsSpan(ParakeetVocabulary.TokenCount, 5));
            if (token != ParakeetVocabulary.Blank)
            {
                state1 = next1; state2 = next2;
                tokens.Add(token); positions.Add(frame); durations.Add(duration);
                emitted++;
            }
            // A positive duration already advances time, including on the last
            // permitted symbol. Do not add a second, spurious frame advance.
            if (duration > 0) { frame += duration; emitted = 0; }
            else if (token == ParakeetVocabulary.Blank || emitted == options.MaxTokensPerFrame)
            { frame++; emitted = 0; }
        }
        return new ParakeetTranscription(vocabulary.Decode(tokens), Array.AsReadOnly(tokens.ToArray()),
            Array.AsReadOnly(positions.ToArray()), Array.AsReadOnly(durations.ToArray()), stop, frames, calls);
    }

    static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++) if (values[i] > values[best]) best = i;
        return best;
    }

    internal static Tensor<float> RequireFloat(IReadOnlyDictionary<string, ITensor> outputs, string name, int[] shape)
    {
        if (!outputs.TryGetValue(name, out var value) || value is not Tensor<float> tensor
            || !tensor.Dimensions.SequenceEqual(shape) || tensor.ToArray().Any(v => !float.IsFinite(v)))
            throw new InvalidDataException("Unexpected or nonfinite Parakeet output: " + name);
        return tensor;
    }
}
