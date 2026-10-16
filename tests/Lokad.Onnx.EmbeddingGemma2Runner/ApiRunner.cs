using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;
using Lokad.Tokenizers.Tokenizer;

internal static class ApiRunner
{
    sealed record Case(string Id, string Role, string Text, string? Title);

    internal static void Run(string[] args)
    {
        if (args.Length != 4) throw new ArgumentException("Usage: runner api prepared-directory cases.json reference-directory output-directory");
        string prepared = args[0], source = args[1], reference = args[2], output = args[3];
        Directory.CreateDirectory(output);
        var options = new EmbeddingGemma2Options { MaximumBatchSize = 3, MaximumPaddedTokens = 1536 };
        var api = new EmbeddingGemma2Embedder(prepared, options);
        var tokenizer = typeof(EmbeddingGemma2Embedder).GetField("tokenizer", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(api)
            as GemmaTokenizer ?? throw new InvalidOperationException("Missing integrated tokenizer");
        using var json = JsonDocument.Parse(File.ReadAllBytes(source));
        var cases = json.RootElement.GetProperty("cases").EnumerateArray().Select(c => new Case(c.GetProperty("id").GetString()!,
            c.GetProperty("role").GetString()!, string.Concat(Enumerable.Repeat(c.GetProperty("text").GetString()!,
                c.TryGetProperty("repeat", out var repeat) ? repeat.GetInt32() : 1)),
            c.TryGetProperty("title", out var title) ? title.GetString() : null)).ToArray();
        var outputs = new Dictionary<string, string>(StringComparer.Ordinal);
        void Write<T>(string relative, T[] values, int[] shape, string descriptor) where T : unmanaged
        {
            string path = Path.Combine(output, relative);
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            Program.WriteNpy(path, values, shape, descriptor);
            outputs.Add(relative.Replace('\\', '/'), Hash(path));
        }
        string Format(Case item) => item.Role switch
        {
            "raw" => item.Text,
            "query" => (string)Call("Query", new object[] { item.Text }),
            "document" => (string)Call("Document", new object[] { new EmbeddingGemma2Document(item.Text, item.Title) }),
            _ => throw new InvalidDataException("Unknown case role"),
        };
        GemmaTokenizedBatch Batch(Func<int, string> text, int offset, int total) =>
            (GemmaTokenizedBatch)Call("Batch", new object[] { text, offset, total, tokenizer, options });
        EmbeddingGemma2Embedding Embed(Case item, int dimensions) => item.Role == "query"
            ? api.EmbedQuery(item.Text, dimensions, CancellationToken.None)
            : api.EmbedDocuments(new[] { new EmbeddingGemma2Document(item.Text, item.Title) }, dimensions, CancellationToken.None)[0];
        var records = new List<object>();
        var spaces = new Dictionary<int, string>();
        foreach (var item in cases)
        {
            var batch = Batch(_ => Format(item), 0, 1);
            var shape = new[] { batch.BatchSize, batch.SequenceLength };
            Write(item.Id + "/input_ids.npy", batch.InputIds, shape, "<i8");
            Write(item.Id + "/attention_mask.npy", batch.AttentionMask, shape, "<i8");
            bool pipeline = item.Role != "raw";
            if (pipeline)
            {
                foreach (int dimension in new[] { 128, 256, 512, 768 })
                {
                    var first = Embed(item, dimension);
                    float[] snapshot = first.Values.ToArray();
                    var repeated = Embed(item, dimension);
                    if (ReferenceEquals(first.Values, repeated.Values) || !SameBytes(snapshot, first.Values) || !SameBytes(snapshot, repeated.Values))
                        throw new InvalidDataException("API ownership/repeatability failed: " + item.Id);
                    if (first.TokenCount != batch.TokenCounts[0] || first.Values.Length != dimension ||
                        spaces.TryGetValue(dimension, out string? expectedSpace) && expectedSpace != first.SpaceId)
                        throw new InvalidDataException("API metadata differs");
                    spaces[dimension] = first.SpaceId;
                    Write(item.Id + $"/embedding_{dimension}.npy", snapshot, new[] { 1, dimension }, "<f4");
                }
            }
            records.Add(new { id = item.Id, api = pipeline });
            Console.WriteLine("PASS integrated API inputs/repeatability " + item.Id);
        }
        var documents = cases.Where(c => c.Role == "document").ToArray();
        var batchRecords = new List<object>();
        for (int offset = 0; offset < documents.Length;)
        {
            var batch = Batch(i => Format(documents[i]), offset, documents.Length);
            string directory = "batches/" + offset;
            var shape = new[] { batch.BatchSize, batch.SequenceLength };
            Write(directory + "/input_ids.npy", batch.InputIds, shape, "<i8");
            Write(directory + "/attention_mask.npy", batch.AttentionMask, shape, "<i8");
            batchRecords.Add(new { directory, ids = documents.Skip(offset).Take(batch.BatchSize).Select(c => c.Id).ToArray(), tokenCounts = batch.TokenCounts });
            offset += batch.BatchSize;
        }
        foreach (int dimension in new[] { 128, 256, 512, 768 })
        {
            var vectors = api.EmbedDocuments(documents.Select(c => new EmbeddingGemma2Document(c.Text, c.Title)).ToArray(), dimension, CancellationToken.None);
            if (vectors.Length != documents.Length) throw new InvalidDataException("API document coverage differs");
            for (int i = 0; i < vectors.Length; i++)
            {
                if (vectors[i].SpaceId != spaces[dimension]) throw new InvalidDataException("Batch space identity differs");
                Write("batched/" + documents[i].Id + $"/embedding_{dimension}.npy", vectors[i].Values, new[] { 1, dimension }, "<f4");
            }
        }
        if (api.EmbedDocuments(Array.Empty<EmbeddingGemma2Document>()).Length != 0) throw new InvalidDataException("Empty request failed");
        Expect<OperationCanceledException>(() => api.EmbedQuery("test", 768, new CancellationToken(true)));
        Expect<ArgumentOutOfRangeException>(() => api.EmbedQuery("test", 100, CancellationToken.None));
        Expect<ArgumentException>(() => api.EmbedDocuments(new[] { new EmbeddingGemma2Document(string.Concat(Enumerable.Repeat("x ", 503)), "Token boundary") }));
        var original = api.EmbedQuery("test");
        float[] before = original.Values.ToArray();
        Array.Fill(original.Values, 0);
        var after = api.EmbedQuery("test");
        if (!SameBytes(before, after.Values)) throw new InvalidDataException("Caller mutation corrupted the next request");
        EmbeddingGemma2Embedding? left = null, right = null;
        Parallel.Invoke(() => left = api.EmbedQuery("test"), () => right = api.EmbedQuery("test"));
        if (left is null || right is null || !SameBytes(before, left.Values) || !SameBytes(before, right.Values) || ReferenceEquals(left.Values, right.Values))
            throw new InvalidDataException("Concurrent callers did not receive independent matching vectors");
        File.WriteAllText(Path.Combine(output, "manifest.json"), JsonSerializer.Serialize(new { schemaVersion = 1, cases = records,
            batches = batchRecords, outputs, spaceIds = spaces, casesSha256 = Hash(source), referenceSha256 = Hash(Path.Combine(reference, "manifest.json")),
            coreSha256 = Hash(typeof(OnnxImport).Assembly.Location), dataSha256 = Hash(typeof(EmbeddingGemma2Embedder).Assembly.Location),
            tokenizerAssemblySha256 = Hash(typeof(GemmaTokenizer).Assembly.Location), tokenizerSha256 = tokenizer.TokenizerSha256,
            runnerSha256 = Hash(typeof(ApiRunner).Assembly.Location), runtime = Environment.Version.ToString(),
            tieredCompilation = Environment.GetEnvironmentVariable("DOTNET_TieredCompilation"),
            tieredPgo = Environment.GetEnvironmentVariable("DOTNET_TieredPGO") }, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine("PASS reused API contexts, batching, ownership, cancellation, overflow, and recovery");
    }

    static object Call(string name, object[] args)
    {
        var type = typeof(EmbeddingGemma2Embedder).Assembly.GetType("Lokad.Onnx.EmbeddingGemma2Inputs", true)!;
        try { return type.GetMethod(name, BindingFlags.NonPublic | BindingFlags.Static)!.Invoke(null, args)!; }
        catch (TargetInvocationException error) when (error.InnerException is not null)
        { ExceptionDispatchInfo.Capture(error.InnerException).Throw(); throw; }
    }

    static void Expect<T>(Action action) where T : Exception
    {
        try { action(); }
        catch (T) { return; }
        throw new InvalidDataException("Expected " + typeof(T).Name);
    }

    static bool SameBytes(float[] a, float[] b) => MemoryMarshal.AsBytes(a.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(b.AsSpan()));
    static string Hash(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }
}
