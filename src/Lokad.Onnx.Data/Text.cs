namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Lokad.Tokenizers.Tokenizer;
using FastBertTokenizer;
using static Lokad.Onnx.Runtime;

public class Text
{
    /// <summary>Loaded BERT tokenizer plus its encode gate.</summary>
    sealed class SharedBertTokenizer
    {
        public BertTokenizer Tokenizer;
        public readonly object Sync = new object();
        public SharedBertTokenizer(BertTokenizer tokenizer) { Tokenizer = tokenizer; }
    }

    static readonly Dictionary<string, SharedBertTokenizer> BertCache = new Dictionary<string, SharedBertTokenizer>();
    static readonly object BertCacheGate = new object();

    /// <summary>Loaded file-backed BERT tokenizer plus its encode gate and observed file identity.</summary>
    sealed class SharedBertFileTokenizer
    {
        public BertTokenizer Tokenizer;
        public readonly object Sync = new object();
        public readonly long FileLength;
        public readonly DateTime FileModifiedUtc;
        public SharedBertFileTokenizer(BertTokenizer tokenizer, long fileLength, DateTime fileModifiedUtc)
        {
            Tokenizer = tokenizer;
            FileLength = fileLength;
            FileModifiedUtc = fileModifiedUtc;
        }
    }

    /// <summary>Process-wide file-backed BERT tokenizer cache keyed by vocab path.</summary>
    /// <remarks>
    /// Lifetime: an entry lives until the file identity (length plus
    /// last-write time) observed at load disagrees with the file on disk,
    /// or the file disappears; there is no time-based eviction. A
    /// disagreed file reloads under the gate, so replaced assets are
    /// never silently stale. A missing file evicts its entry and throws
    /// like a never-loaded path. Fully offline: no entry here ever
    /// fetches. Concurrency and ownership match the Roberta cache:
    /// lookup, load, and reload serialize on the gate, which is never
    /// held during encoding, and encodes serialize on the per-instance
    /// gate through the Text methods.
    /// </remarks>
    static readonly Dictionary<string, SharedBertFileTokenizer> BertFileCache = new Dictionary<string, SharedBertFileTokenizer>();
    static readonly object BertFileCacheGate = new object();

    /// <summary>Loads (downloading when absent) and caches the named BERT tokenizer.</summary>
    /// <remarks>Shared process-wide like the Roberta cache. The async load runs
    /// on a worker thread so waiting for it cannot deadlock a captured
    /// synchronization context. BERT stays outside the offline guarantee:
    /// this entry performs network acquisition.</remarks>
    public static BertTokenizer GetOrLoadBertTokenizer(string tokenizer)
    {
        lock (BertCacheGate)
        {
            if (BertCache.TryGetValue(tokenizer, out var cached)) return cached.Tokenizer;
            var tok = new BertTokenizer();
            System.Threading.Tasks.Task.Run(() => tok.LoadFromHuggingFaceAsync(tokenizer)).GetAwaiter().GetResult();
            BertCache[tokenizer] = new SharedBertTokenizer(tok);
            return tok;
        }
    }

    /// <summary>Tokenizes one text with a BERT tokenizer.</summary>
    /// <remarks>Uses the shared cached instance for the id; encodes serialize
    /// on that instance gate. Requires network on first use per id.</remarks>
    public static ITensor[]? BertTokenize(string text, string tokenizer)
    {
        var op = Begin("Tokenizing {len} characters using BERT tokenizer {tok}", text.Length, tokenizer);
        SharedBertTokenizer shared;
        lock (BertCacheGate)
        {
            if (!BertCache.TryGetValue(tokenizer, out var cached))
            {
                var tok = new BertTokenizer();
                System.Threading.Tasks.Task.Run(() => tok.LoadFromHuggingFaceAsync(tokenizer)).GetAwaiter().GetResult();
                cached = new SharedBertTokenizer(tok);
                BertCache[tokenizer] = cached;
            }
            shared = cached;
        }
        ITensor[] result;
        lock (shared.Sync)
        {
            var (eInputIds, eAttentionMask, eTokenTypeIds) = shared.Tokenizer.Encode(text, 512);
            result = new ITensor[3] {
                DenseTensor<long>.OfValues(eInputIds.ToArray()).PadLeft().WithName("input_ids"),
                DenseTensor<long>.OfValues(eAttentionMask.ToArray()).PadLeft().WithName("attention_mask"),
                DenseTensor<long>.OfValues(eTokenTypeIds.ToArray()).PadLeft().WithName("token_type_ids"),
            };
        }
        op.Complete();
        return result;
    }

    /// <summary>Loads a BERT tokenizer from a local WordPiece vocab file without network access.</summary>
    /// <remarks>Input is not lowercased; the vocab file decides casing. Throws
    /// FileNotFoundException for a missing path.</remarks>
    public static BertTokenizer LoadBertTokenizerFromFile(string vocabPath)
    {
        if (!File.Exists(vocabPath))
        {
            throw new FileNotFoundException($"Tokenizer vocab file does not exist: {vocabPath}.", vocabPath);
        }
        var tok = new BertTokenizer();
        using var reader = new StreamReader(vocabPath);
        tok.LoadVocabulary(reader, false, "[UNK]", "[CLS]", "[SEP]", "[PAD]", System.Text.NormalizationForm.FormC);
        return tok;
    }

    /// <summary>Returns the shared cached holder for a vocab file, loading it on first use.</summary>
    /// <remarks>A cached entry whose file identity disagrees reloads here, and a
    /// deleted file evicts its entry and throws. The gate is never held
    /// during encoding.</remarks>
    static SharedBertFileTokenizer GetOrLoadBertFileTokenizerLocked(string vocabPath)
    {
        lock (BertFileCacheGate)
        {
            var identity = TokenizerFileIdentity(vocabPath);
            if (identity is null)
            {
                BertFileCache.Remove(vocabPath);
                throw new FileNotFoundException($"Tokenizer vocab file does not exist: {vocabPath}.", vocabPath);
            }
            if (!BertFileCache.TryGetValue(vocabPath, out var cached)
                || cached.FileLength != identity.Value.Length
                || cached.FileModifiedUtc != identity.Value.ModifiedUtc)
            {
                cached = new SharedBertFileTokenizer(
                    LoadBertTokenizerFromFile(vocabPath),
                    identity.Value.Length,
                    identity.Value.ModifiedUtc);
                BertFileCache[vocabPath] = cached;
            }
            return cached;
        }
    }

    /// <summary>Returns the shared cached file-backed BERT tokenizer for a vocab path.</summary>
    /// <remarks>Fully offline. Encodes through the returned instance are only
    /// safe via the Text encode methods, which serialize on the instance
    /// gate. A missing path throws FileNotFoundException and evicts any entry.</remarks>
    public static BertTokenizer GetOrLoadBertFileTokenizer(string vocabPath)
    {
        return GetOrLoadBertFileTokenizerLocked(vocabPath).Tokenizer;
    }

    /// <summary>Tokenizes one text with the shared cached file-backed BERT tokenizer.</summary>
    /// <remarks>Fully offline: reuses the cached instance and never downloads.
    /// Encodes serialize on the instance gate.</remarks>
    public static ITensor[]? BertTokenizeFromFile(string text, string vocabPath)
    {
        var op = Begin("Tokenizing {len} characters using BERT vocab file {f}", text.Length, vocabPath);
        var shared = GetOrLoadBertFileTokenizerLocked(vocabPath);
        ITensor[] result;
        lock (shared.Sync)
        {
            var (eInputIds, eAttentionMask, eTokenTypeIds) = shared.Tokenizer.Encode(text, 512);
            result = new ITensor[3] {
                DenseTensor<long>.OfValues(eInputIds.ToArray()).PadLeft().WithName("input_ids"),
                DenseTensor<long>.OfValues(eAttentionMask.ToArray()).PadLeft().WithName("attention_mask"),
                DenseTensor<long>.OfValues(eTokenTypeIds.ToArray()).PadLeft().WithName("token_type_ids"),
            };
        }
        op.Complete();
        return result;
    }

    public static XLMRobertaTokenizer LoadRobertaTokenizerFromFile(string tokenizerModelPath)
    {
        if (!File.Exists(tokenizerModelPath))
        {
            throw new FileNotFoundException($"Tokenizer model file does not exist: {tokenizerModelPath}.", tokenizerModelPath);
        }
        return new XLMRobertaTokenizer(tokenizerModelPath, false);
    }

    static string NormalizeRobertaText(string text) => System.Text.RegularExpressions.Regex.Replace(text, "  +", " ");

    static ITensor[]? EncodeSingleRoberta(XLMRobertaTokenizer tok, string text, string tokDesc)
    {
        var op = Begin("Tokenizing text of length {l} chars using {tok_desc} tokenizer", text.Length, tokDesc);
        var t = tok.Encode(NormalizeRobertaText(text), null, 512, TruncationStrategy.OnlyFirst, 0);
        if (t is null)
        {
            op.Abandon();
            return null;
        }
        else
        {
            op.Complete();
            return new ITensor[3]
            {
                 DenseTensor<long>.OfValues(t.TokenIds.ToArray()).PadLeft().WithName("input_ids"),
                 DenseTensor<long>.Ones(1, t.Mask.Count).WithName("attention_mask"),
                 DenseTensor<long>.Zeros(1, t.TokenIds.Count).WithName("token_type_ids"),
            };
        }
    }

    static ITensor[] EncodeBatchRoberta(XLMRobertaTokenizer tok, string[] texts, string tokDesc)
    {
        var op = Begin("Tokenizing text array of length {l} using {tok_desc} tokenizer", texts.Length, tokDesc);
        // Single pass: each text is encoded exactly once into a materialized
        // id array while tracking the maximum length inline. The three batch
        // tensors are then allocated once and filled directly: input ids copy
        // each row and pad-fill with 1, attention writes ones over real tokens
        // leaving pad zeros, and type ids stay zero-initialized.
        var ids = new List<long[]>(texts.Length);
        int maxl = 0;
        foreach (var text1 in texts)
        {
            var t = tok.Encode(NormalizeRobertaText(text1), null, 512, TruncationStrategy.OnlyFirst, 0);
            if (t is null)
            {
                op.Abandon();
                throw new Exception("Error tokenizing text " + text1 + ". Stopping.");
            }
            var arr = t.TokenIds.ToArray();
            ids.Add(arr);
            if (arr.Length > maxl) maxl = arr.Length;
        }
        if (ids.Count == 0)
        {
            op.Complete();
            return new ITensor[] {
                new DenseTensor<long>(new Memory<long>(Array.Empty<long>()), new[]{0, 0}).WithName("input_ids"),
                new DenseTensor<long>(new Memory<long>(Array.Empty<long>()), new[]{0, 0}).WithName("attention_mask"),
                new DenseTensor<long>(new Memory<long>(Array.Empty<long>()), new[]{0, 0}).WithName("token_type_ids")
            };
        }
        var inputIds = new DenseTensor<long>(new Memory<long>(new long[ids.Count * maxl]), new[]{ids.Count, maxl});
        var attentionMask = new DenseTensor<long>(new Memory<long>(new long[ids.Count * maxl]), new[]{ids.Count, maxl});
        var typeIds = new DenseTensor<long>(new Memory<long>(new long[ids.Count * maxl]), new[]{ids.Count, maxl});
        var ii = inputIds.Buffer.Span;
        var am = attentionMask.Buffer.Span;
        for (int i = 0; i < ids.Count; i++)
        {
            var row = ids[i];
            row.CopyTo(ii.Slice(i * maxl, row.Length));
            ii.Slice(i * maxl + row.Length, maxl - row.Length).Fill(1L);
            am.Slice(i * maxl, row.Length).Fill(1L);
        }
        op.Complete();
        return new ITensor[] {
            inputIds.WithName("input_ids"),
            attentionMask.WithName("attention_mask"),
            typeIds.WithName("token_type_ids")
        };
    }

    /// <summary>Loaded Roberta tokenizer plus its encode gate and observed file identity.</summary>
    sealed class SharedRobertaTokenizer
    {
        public XLMRobertaTokenizer Tokenizer;
        public readonly object Sync = new object();
        public readonly long FileLength;
        public readonly DateTime FileModifiedUtc;
        public SharedRobertaTokenizer(XLMRobertaTokenizer tokenizer, long fileLength, DateTime fileModifiedUtc)
        {
            Tokenizer = tokenizer;
            FileLength = fileLength;
            FileModifiedUtc = fileModifiedUtc;
        }
    }

    /// <summary>Process-wide Roberta tokenizer cache keyed by model path.</summary>
    /// <remarks>
    /// Lifetime: an entry lives until the file identity (length plus
    /// last-write time) observed at load disagrees with the file on disk,
    /// or the file disappears; there is no time-based eviction. A
    /// disagreed file reloads under the gate, so replaced assets are
    /// never silently stale. A missing file evicts its entry and throws
    /// like a never-loaded path. Concurrency: lookup, load, and reload
    /// serialize on the gate, which is never held during encoding;
    /// encodes serialize on the per-instance gate instead. Ownership:
    /// instances are shared. Encoding through the Text methods is
    /// synchronized; calling Encode on a returned instance directly from
    /// several threads is not.
    /// </remarks>
    static readonly Dictionary<string, SharedRobertaTokenizer> RobertaCache = new Dictionary<string, SharedRobertaTokenizer>();
    static readonly object RobertaCacheGate = new object();

    const string Me5sTokenizerFileName = "me5s-sentencepiece.bpe.model";

    /// <summary>
    /// Searches a directory and its ancestors for a relative file path,
    /// returning the first existing file or null.
    /// </summary>
    public static string? FindAssetUnderAncestors(string startDirectory, params string[] parts)
    {
        if (parts.Length == 0) return null;
        var relative = Path.Combine(parts);
        var dir = new DirectoryInfo(startDirectory);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, relative);
            if (File.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        return null;
    }

    /// <summary>
    /// Resolved usable path for the bundled multilingual-e5-small tokenizer:
    /// the binary cache when present, otherwise the documented models tree,
    /// otherwise the cache path as the acquisition target.
    /// </summary>
    public static string Me5sTokenizerPath()
    {
        var cached = Path.Combine(AssemblyLocation, Me5sTokenizerFileName);
        if (File.Exists(cached)) return cached;
        var documented = FindAssetUnderAncestors(
            Directory.GetCurrentDirectory(), "models", "multilingual-e5-small", "sentencepiece.bpe.model");
        if (documented is not null) return documented;
        return cached;
    }

    static bool DownloadAsset(string name, Uri downloadUrl, string downloadPath)
    {
        using var op = Begin("Downloading {0} from {1} to {2}", name, downloadUrl, downloadPath);
        try
        {
            if (File.Exists(downloadPath)) Warn("File {0} exists, overwriting...", downloadPath);
            using var client = new System.Net.Http.HttpClient() { Timeout = TimeSpan.FromMinutes(10) };
            var bytes = client.GetByteArrayAsync(downloadUrl).GetAwaiter().GetResult();
            File.WriteAllBytes(downloadPath, bytes);
        }
        catch (Exception ex)
        {
            Error(ex, "Could not download {0} from {1}.", name, downloadUrl);
            return false;
        }
        if (File.Exists(downloadPath))
        {
            op.Complete();
            return true;
        }
        Error("Did not locate file at {p}.", downloadPath);
        return false;
    }

    /// <summary>Acquires the me5s tokenizer asset, downloading it when absent everywhere.</summary>
    /// <remarks>This is the only tokenizer entry that performs network
    /// acquisition; all encoding entries are offline and fail clearly when
    /// their asset is missing. Returns false when the asset cannot be obtained.</remarks>
    public static bool EnsureMe5sTokenizer()
    {
        var resolved = Me5sTokenizerPath();
        if (File.Exists(resolved)) return true;
        if (!DownloadAsset(
            "sentencepiece.bpe.model",
            new Uri("https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/sentencepiece.bpe.model"),
            Path.Combine(AssemblyLocation, Me5sTokenizerFileName)))
        {
            Error("Could not download model file.");
            return false;
        }
        return true;
    }

    /// <summary>Tokenizes one text with the named Roberta tokenizer.</summary>
    /// <remarks>Offline: a missing me5s asset fails clearly instead of
    /// downloading; call EnsureMe5sTokenizer first when acquisition is
    /// wanted. Encodes serialize on the shared instance gate.</remarks>
    public static ITensor[]? RobertaTokenize(string text1, string tokenizer)
    {
        switch (tokenizer)
        {
            case "me5s":
                var tokenizerPath = Me5sTokenizerPath();
                if (!File.Exists(tokenizerPath))
                {
                    Error("Tokenizer asset {f} is missing; call EnsureMe5sTokenizer to acquire it.", tokenizerPath);
                    return null;
                }
                var shared = GetOrLoadRobertaTokenizerLocked(tokenizerPath);
                lock (shared.Sync)
                {
                    return EncodeSingleRoberta(shared.Tokenizer, text1, "multilingual-e5-small");
                }
            default:
                Error("Unknown Roberta tokenizer: {t}.", tokenizer);
                return null;
        }
    }

    /// <summary>Tokenizes a batch of texts with the named Roberta tokenizer.</summary>
    /// <remarks>Offline: a missing me5s asset fails clearly instead of
    /// downloading; call EnsureMe5sTokenizer first when acquisition is
    /// wanted. Encodes serialize on the shared instance gate.</remarks>
    public static ITensor[]? RobertaTokenize(string[] text, string tokenizer)
    {
        switch (tokenizer)
        {
            case "me5s":
                var tokenizerPath = Me5sTokenizerPath();
                if (!File.Exists(tokenizerPath))
                {
                    Error("Tokenizer asset {f} is missing; call EnsureMe5sTokenizer to acquire it.", tokenizerPath);
                    return null;
                }
                var shared = GetOrLoadRobertaTokenizerLocked(tokenizerPath);
                lock (shared.Sync)
                {
                    return EncodeBatchRoberta(shared.Tokenizer, text, "multilingual-e5-small");
                }
            default:
                Error("Unknown Roberta tokenizer: {t}.", tokenizer);
                return null;
        }
    }

    /// <summary>Returns the shared cached holder, loading it on first use.</summary>
    /// <remarks>Only the dictionary membership is gated; the load itself runs
    /// inside the same gate so two threads never load the same path twice.
    /// A cached entry whose file identity disagrees reloads here, and a
    /// deleted file evicts its entry and throws. The gate is never held
    /// during encoding.</remarks>
    static SharedRobertaTokenizer GetOrLoadRobertaTokenizerLocked(string tokenizerModelPath)
    {
        lock (RobertaCacheGate)
        {
            var identity = TokenizerFileIdentity(tokenizerModelPath);
            if (identity is null)
            {
                RobertaCache.Remove(tokenizerModelPath);
                throw new FileNotFoundException($"Tokenizer model file does not exist: {tokenizerModelPath}.", tokenizerModelPath);
            }
            if (!RobertaCache.TryGetValue(tokenizerModelPath, out var cached)
                || cached.FileLength != identity.Value.Length
                || cached.FileModifiedUtc != identity.Value.ModifiedUtc)
            {
                cached = new SharedRobertaTokenizer(
                    LoadRobertaTokenizerFromFile(tokenizerModelPath),
                    identity.Value.Length,
                    identity.Value.ModifiedUtc);
                RobertaCache[tokenizerModelPath] = cached;
            }
            return cached;
        }
    }

    /// <summary>Reads the reload identity of a tokenizer model file.</summary>
    /// <remarks>Null means absent, which the caller reports exactly like a
    /// never-loaded missing path instead of serving a stale instance.</remarks>
    static (long Length, DateTime ModifiedUtc)? TokenizerFileIdentity(string tokenizerModelPath)
    {
        if (string.IsNullOrEmpty(tokenizerModelPath)) return null;
        var info = new FileInfo(tokenizerModelPath);
        if (!info.Exists) return null;
        return (info.Length, info.LastWriteTimeUtc);
    }

    /// <summary>Returns the shared cached Roberta tokenizer for a model path.</summary>
    /// <remarks>Shared process-wide; entries reload when the file identity
    /// changes and evict when the file disappears (see the cache remarks).
    /// Encodes through the returned instance are only safe via the Text
    /// encode methods, which serialize on the instance gate; calling Encode
    /// on the instance directly from several threads is not synchronized.
    /// A missing path throws FileNotFoundException and evicts any entry.</remarks>
    public static XLMRobertaTokenizer GetOrLoadRobertaTokenizer(string tokenizerModelPath)
    {
        return GetOrLoadRobertaTokenizerLocked(tokenizerModelPath).Tokenizer;
    }

    /// <summary>Tokenizes one text with the shared cached file tokenizer.</summary>
    /// <remarks>Fully offline: reuses the cached instance and never downloads.
    /// Encodes serialize on the instance gate.</remarks>
    public static ITensor[]? RobertaTokenizeFromFile(string text, string tokenizerModelPath)
    {
        var shared = GetOrLoadRobertaTokenizerLocked(tokenizerModelPath);
        lock (shared.Sync)
        {
            return EncodeSingleRoberta(shared.Tokenizer, text, "multilingual-e5-small");
        }
    }

    /// <summary>Tokenizes a batch of texts with the shared cached file tokenizer.</summary>
    /// <remarks>Fully offline: reuses the cached instance and never downloads.
    /// Encodes serialize on the instance gate.</remarks>
    public static ITensor[]? RobertaTokenizeFromFile(IReadOnlyList<string> texts, string tokenizerModelPath)
    {
        var shared = GetOrLoadRobertaTokenizerLocked(tokenizerModelPath);
        lock (shared.Sync)
        {
            return EncodeBatchRoberta(shared.Tokenizer, texts.ToArray(), "multilingual-e5-small");
        }
    }
    public static ITensor[]? GetTextTensors(string text, string props)
    {
        // An empty properties string selects the default me5s tokenizer:
        // Split never yields an empty array, so test the head instead.
        var tprops = props.Split(':');
        if (tprops.Length == 0 || string.IsNullOrEmpty(tprops[0]) || tprops[0] == "me5s")
        {
            return RobertaTokenize(text, "me5s");
        }
        else if (tprops[0] == "bert")
        {
            return BertTokenize(text, "bert-base-uncased");
        }
        else
        {
            Error("Could not tokenize text using properties {p}.", props);
            return null;
        }
    }

    public static ITensor[]? GetTextTensors(string[] text, string props)
    {
        var tprops = props.Split(':');
        if (tprops.Length == 0 || string.IsNullOrEmpty(tprops[0]) || tprops[0] == "me5s")
        {
            return RobertaTokenize(text, "me5s");
        }
        //else if (tprops[0] == "bert")
        //{
        //    return BertTokenize(text, "bert-base-uncased");
        //}
        else
        {
            Error("Could not tokenize text using properties {p}.", props);
            return null;
        }
    }

    public static ITensor[]? GetTextTensorsFromFileArg(string name, string[] p)
    {
        if (!File.Exists(name)) 
        {
            Error("File {name} does not exist.", name); 
            return null;
        }
        else
        {
            return GetTextTensors(File.ReadAllText(name), p.Length > 0 ? p[0] : "");
        }
    }

    public static string[] TextExtensions = new string[] { ".txt" };
}

