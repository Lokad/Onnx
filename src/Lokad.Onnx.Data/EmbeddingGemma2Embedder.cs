namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading;
using Lokad.Tokenizers.Tokenizer;

/// <summary>A document for the model's retrieval prompt. A null or empty title becomes "none".</summary>
public sealed record EmbeddingGemma2Document(string Text, string? Title)
{
    /// <summary>Create a document without a title.</summary>
    public EmbeddingGemma2Document(string text) : this(text, null) { }
}

/// <summary>An owned unit vector, its real token count, and its embedding-space identity.</summary>
public sealed record EmbeddingGemma2Embedding(float[] Values, int TokenCount, string SpaceId);

/// <summary>Offline, managed FP32 text embeddings from the prepared EmbeddingGemma 2 export.</summary>
/// <remarks>One instance loads assets once and serializes calls. Cancellation is checked between
/// batches; an in-flight graph evaluation completes first. Returned vectors own their storage.
/// The default 512-token limit includes prompts and special tokens. Larger limits require separate
/// sequence-length and resource qualification; the model's advertised ceiling is 8192 tokens.</remarks>
public sealed class EmbeddingGemma2Embedder
{
    const string SourceGraph = "bc47de15f81208a5c99e5ab10f746d5e33b51ea228b7dc0bef9c133a94f1c1c3";
    const string Weights = "9fd452bfc6916e5a92f080f8dada35598ca4512b21c65897d4c5974d5f9630a2";
    const string Tokenizer = "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4";
    readonly GraphExecution execution;
    readonly GemmaTokenizer tokenizer;
    readonly EmbeddingGemma2Options options;
    readonly string identity;
    readonly object gate = new object();

    /// <summary>Load a local prepared bundle with the default inference bounds.</summary>
    public EmbeddingGemma2Embedder(string preparedDirectory) : this(preparedDirectory, new EmbeddingGemma2Options()) { }

    /// <summary>Validate prepared graph, FP32 weights and tokenizer identities, then load once.</summary>
    public EmbeddingGemma2Embedder(string preparedDirectory, EmbeddingGemma2Options options)
    {
        ArgumentException.ThrowIfNullOrEmpty(preparedDirectory);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        this.options = options;
        using var receipt = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(preparedDirectory, "preparation.json")));
        var root = receipt.RootElement;
        string modelPath = Path.Combine(preparedDirectory, "model.onnx"), graphHash = Hash(modelPath);
        if (root.GetProperty("schemaVersion").GetInt32() != 1 || root.GetProperty("precision").GetString() != "fp32" ||
            root.GetProperty("sourceGraphSha256").GetString() != SourceGraph || root.GetProperty("graphSha256").GetString() != graphHash ||
            root.GetProperty("tokenizerSha256").GetString() != Tokenizer || root.GetProperty("weightsSha256").GetString() != Weights ||
            Hash(Path.Combine(preparedDirectory, "model.onnx_data")) != Weights)
            throw new InvalidDataException("The prepared bundle is not the pinned FP32 EmbeddingGemma 2 export.");
        tokenizer = new GemmaTokenizer(Path.Combine(preparedDirectory, "tokenizer.json"));
        if (tokenizer.TokenizerSha256 != Tokenizer) throw new InvalidDataException("Tokenizer identity differs from the prepared bundle.");
        var graph = OnnxImport.Load(modelPath, options.PreparedWeightBudgetBytes)
            ?? throw new InvalidDataException(OnnxImport.LastErrorMessage, OnnxImport.LastErrorCause);
        if (graph.Inputs.Count != 2 || !graph.Inputs.ContainsKey("input_ids") || !graph.Inputs.ContainsKey("attention_mask") ||
            graph.Outputs.Count != 1 || !graph.Outputs.ContainsKey("last_hidden_state"))
            throw new InvalidDataException("Expected the prepared text-only graph with projected hidden-state output.");
        execution = graph.CreateExecution(ExecutionOptions.Memory);
        identity = SourceGraph + ":" + graphHash + ":" + Weights + ":" + Tokenizer + ":" +
            root.GetProperty("preparerSha256").GetString() + ":retrieval-prompts-v1:fp32:masked-mean:truncate-then-l2";
    }

    /// <summary>Embed a query at the full 768 dimensions.</summary>
    public EmbeddingGemma2Embedding EmbedQuery(string query) => EmbedQuery(query, 768, CancellationToken.None);

    /// <summary>Embed a retrieval query at one of the supported dimensions.</summary>
    public EmbeddingGemma2Embedding EmbedQuery(string query, int dimensions, CancellationToken cancellation)
    {
        string formatted = EmbeddingGemma2Inputs.Query(query);
        return Embed(_ => formatted, 1, dimensions, cancellation)[0];
    }

    /// <summary>Embed documents at the full 768 dimensions, preserving order.</summary>
    public EmbeddingGemma2Embedding[] EmbedDocuments(IReadOnlyList<EmbeddingGemma2Document> documents) =>
        EmbedDocuments(documents, 768, CancellationToken.None);

    /// <summary>Embed documents in bounded batches, rejecting token overflow without truncation.</summary>
    public EmbeddingGemma2Embedding[] EmbedDocuments(IReadOnlyList<EmbeddingGemma2Document> documents,
        int dimensions, CancellationToken cancellation)
    {
        ArgumentNullException.ThrowIfNull(documents);
        return Embed(i => EmbeddingGemma2Inputs.Document(documents[i]), documents.Count, dimensions, cancellation);
    }

    EmbeddingGemma2Embedding[] Embed(Func<int, string> text, int total, int dimensions, CancellationToken cancellation)
    {
        EmbeddingGemma2Pooling.ValidateDimensions(dimensions);
        cancellation.ThrowIfCancellationRequested();
        string spaceId = "embeddinggemma2:" + Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(identity + ":" + dimensions.ToString(CultureInfo.InvariantCulture))));
        lock (gate)
        {
            cancellation.ThrowIfCancellationRequested();
            var results = new EmbeddingGemma2Embedding[total];
            int offset = 0;
            while (offset < total)
            {
                cancellation.ThrowIfCancellationRequested();
                var batch = EmbeddingGemma2Inputs.Batch(text, offset, total, tokenizer, options);
                var shape = new[] { batch.BatchSize, batch.SequenceLength };
                var ids = new DenseTensor<long>(batch.InputIds, shape);
                var mask = new DenseTensor<long>(batch.AttentionMask, shape);
                try
                {
                    if (!execution.Execute(new Dictionary<string, ITensor> { ["input_ids"] = ids, ["attention_mask"] = mask },
                        true, ExecutionProvider.CPU, ExecutionOptions.Memory))
                        throw new InvalidDataException(execution.LastErrorMessage, execution.LastErrorCause);
                    if (execution.Outputs["last_hidden_state"] is not Tensor<float> hidden)
                        throw new InvalidDataException("Missing projected FP32 hidden states.");
                    var vectors = EmbeddingGemma2Pooling.Pool(hidden, mask, dimensions, cancellation);
                    for (int b = 0; b < batch.BatchSize; b++)
                        results[offset + b] = new EmbeddingGemma2Embedding(vectors[b], batch.TokenCounts[b], spaceId);
                    offset += batch.BatchSize;
                }
                finally { execution.Reset(); }
            }
            cancellation.ThrowIfCancellationRequested();
            return results;
        }
    }

    static string Hash(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }
}
