namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using Lokad.Tokenizers.Tokenizer;

internal static class EmbeddingGemma2Inputs
{
    internal static string Query(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return "task: search result | query: " + text;
    }

    internal static string Document(EmbeddingGemma2Document document)
    {
        ArgumentNullException.ThrowIfNull(document);
        ArgumentNullException.ThrowIfNull(document.Text);
        return "title: " + (string.IsNullOrEmpty(document.Title) ? "none" : document.Title) + " | text: " + document.Text;
    }

    internal static GemmaTokenizedBatch Batch(Func<int, string> text, int offset, int total,
        GemmaTokenizer tokenizer, EmbeddingGemma2Options options)
    {
        var rows = new List<long[]>();
        int width = 0;
        for (int i = offset; i < total && rows.Count < options.MaximumBatchSize; i++)
        {
            long[] ids = tokenizer.Encode(text(i));
            if (ids.Length > options.MaximumTokens)
                throw new ArgumentException($"Input {i} requires {ids.Length} tokens; limit is {options.MaximumTokens} including prefix and special tokens.");
            int nextWidth = Math.Max(width, ids.Length);
            if ((long)(rows.Count + 1) * nextWidth > options.MaximumPaddedTokens) break;
            rows.Add(ids);
            width = nextWidth;
        }
        if (rows.Count == 0) throw new ArgumentException("The padded-token budget cannot accommodate an input.");
        var values = new long[checked(rows.Count * width)];
        var mask = new long[values.Length];
        var counts = new int[rows.Count];
        for (int i = 0; i < rows.Count; i++)
        {
            rows[i].CopyTo(values, i * width);
            mask.AsSpan(i * width, rows[i].Length).Fill(1);
            counts[i] = rows[i].Length;
        }
        return new GemmaTokenizedBatch(values, mask, counts, rows.Count, width);
    }
}
