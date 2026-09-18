namespace Lokad.Onnx;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;

// The decoder consumes byte-level BPE tokens. Encoding and text normalization
// are not needed to turn generated token IDs into their original UTF-8 text.
internal sealed class WhisperTokenizer
{
    readonly Dictionary<int, byte[]> tokens = new Dictionary<int, byte[]>();
    readonly Dictionary<string, int> special = new Dictionary<string, int>(StringComparer.Ordinal);

    internal WhisperTokenizer(Stream stream)
    {
        using var document = JsonDocument.Parse(stream);
        var root = document.RootElement;
        if (root.GetProperty("model").GetProperty("type").GetString() != "BPE"
            || root.GetProperty("decoder").GetProperty("type").GetString() != "ByteLevel")
            throw new InvalidDataException("Whisper requires a byte-level BPE tokenizer.json.");
        var reverse = new Dictionary<char, byte>();
        int extra = 256;
        for (int b = 0; b < 256; b++)
        {
            bool literal = b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174;
            reverse.Add((char)(literal ? b : extra++), (byte)b);
        }
        foreach (var token in root.GetProperty("model").GetProperty("vocab").EnumerateObject())
        {
            var bytes = new byte[token.Name.Length];
            for (int i = 0; i < bytes.Length; i++)
                if (!reverse.TryGetValue(token.Name[i], out bytes[i]))
                    throw new InvalidDataException("Vocabulary contains a non-byte-level token.");
            if (token.Value.GetInt32() < 0 || !tokens.TryAdd(token.Value.GetInt32(), bytes))
                throw new InvalidDataException("Vocabulary has an invalid or duplicate token ID.");
        }
        foreach (var token in root.GetProperty("added_tokens").EnumerateArray())
        {
            string content = token.GetProperty("content").GetString()
                ?? throw new InvalidDataException("Missing added token text.");
            int id = token.GetProperty("id").GetInt32();
            if (id < 0 || !special.TryAdd(content, id))
                throw new InvalidDataException("Invalid or duplicate added token.");
        }
    }

    internal int SpecialId(string content) => special.TryGetValue(content, out int id)
        ? id : throw new InvalidDataException("Tokenizer is missing " + content);

    internal string Decode(IEnumerable<int> ids, int endToken)
    {
        var bytes = new ArrayBufferWriter<byte>();
        foreach (int id in ids)
        {
            if (id == endToken) break;
            if (!tokens.TryGetValue(id, out var token))
                throw new InvalidDataException("Generated token is not a text token: " + id);
            token.CopyTo(bytes.GetSpan(token.Length));
            bytes.Advance(token.Length);
        }
        // Decode after joining bytes: a UTF-8 character can span several tokens.
        // Replacement fallback agrees with the tokenizer's incomplete-byte behavior.
        return Encoding.UTF8.GetString(bytes.WrittenSpan);
    }
}
