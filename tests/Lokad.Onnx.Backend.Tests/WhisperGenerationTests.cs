namespace Lokad.Onnx.Backend.Tests;

using System.Text;
using System.Text.Json;
using System.Reflection;
using System.Runtime.ExceptionServices;

public class WhisperGenerationTests
{
    // Keep policy/tokenizer implementation internal without adding friend assemblies.
    sealed class WhisperTokenizer
    {
        internal readonly object Value;
        internal WhisperTokenizer(Stream stream) => Value = Construct("WhisperTokenizer", stream);
        internal string Decode(IEnumerable<int> ids, int end) => (string)Invoke(Value, "Decode", ids, end);
    }
    sealed class WhisperGeneration
    {
        readonly object value;
        internal WhisperGeneration(Stream stream, WhisperTokenizer tokenizer) => value = Construct("WhisperGeneration", stream, tokenizer.Value);
        internal void Validate(WhisperTranscriptionOptions options) => Invoke(value, "Validate", options);
        internal WhisperTranscription Decode(Tensor<float> hidden, WhisperTranscriptionOptions options,
            Func<bool, Dictionary<string, ITensor>, IReadOnlyDictionary<string, ITensor>> execute, CancellationToken cancellation)
            => (WhisperTranscription)Invoke(value, "Decode", hidden, options, execute, cancellation);
    }
    static object Construct(string name, params object[] args)
    {
        var type = typeof(WhisperTranscriber).Assembly.GetType("Lokad.Onnx." + name, true)!;
        return Unwrap(() => Activator.CreateInstance(type, BindingFlags.Instance | BindingFlags.NonPublic, null, args, null)!);
    }
    static object Invoke(object target, string method, params object[] args) => Unwrap(() =>
        target.GetType().GetMethod(method, BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(target, args)!);
    static object Unwrap(Func<object> call)
    {
        try { return call(); }
        catch (TargetInvocationException e) when (e.InnerException is not null)
        { ExceptionDispatchInfo.Capture(e.InnerException).Throw(); throw; }
    }

    const int Vocab = 51866, Eos = 50257;
    static JsonDocument Fixture() => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "whisper-text.json")));
    static MemoryStream Stream(JsonElement value) => new MemoryStream(Encoding.UTF8.GetBytes(value.GetRawText()));
    static (WhisperTokenizer, WhisperGeneration) Create()
    {
        using var fixture = Fixture();
        using var tokenizerData = Stream(fixture.RootElement.GetProperty("tokenizer"));
        using var configData = Stream(fixture.RootElement.GetProperty("generation"));
        var tokenizer = new WhisperTokenizer(tokenizerData);
        return (tokenizer, new WhisperGeneration(configData, tokenizer));
    }
    static DenseTensor<float> Hidden() => new DenseTensor<float>(new[] { 1, 2, 1280 });

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)]
    [InlineData(4)] [InlineData(5)] [InlineData(6)] [InlineData(7)]
    [InlineData(8)] [InlineData(9)] [InlineData(10)] [InlineData(11)]
    public void DecodeMatchesIndependentTokenizerIncludingSplitUtf8(int index)
    {
        var (tokenizer, _) = Create();
        using var fixture = Fixture();
        var row = fixture.RootElement.GetProperty("cases")[index];
        Assert.Equal(row.GetProperty("text").GetString(), tokenizer.Decode(row.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()), Eos));
    }

    [Fact]
    public void UnknownTextTokenIsRejected()
    {
        var (tokenizer, _) = Create();
        Assert.Throws<InvalidDataException>(() => tokenizer.Decode(new[] { -1 }, Eos));
        Assert.Throws<InvalidDataException>(() => tokenizer.Decode(new[] { 50363 }, Eos));
    }

    static Dictionary<string, ITensor> Outputs(int step, int chosen, bool noSpeech, bool confident)
    {
        int length = step == 0 ? 4 : 1;
        var logits = Enumerable.Repeat(confident ? -20f : -2f, length * Vocab).ToArray();
        logits[(length - 1) * Vocab + chosen] = confident ? 20 : 0;
        if (step == 0) logits[50363] = noSpeech ? 30 : -30;
        var result = new Dictionary<string, ITensor> { ["logits"] = new DenseTensor<float>(logits, new[] { 1, length, Vocab }) };
        for (int layer = 0; layer < 4; layer++)
            foreach (string kind in new[] { "key", "value" })
            {
                result.Add($"present.{layer}.decoder.{kind}", new DenseTensor<float>(new[] { 1, 20, 4 + step, 64 }));
                if (step == 0) result.Add($"present.{layer}.encoder.{kind}", new DenseTensor<float>(new[] { 1, 20, 2, 64 }));
            }
        return result;
    }

    [Theory]
    [InlineData("en", 50259)]
    [InlineData("fr", 50265)]
    public void RequestsUseTheirOwnCachesAndCarryOriginalCrossAttention(string language, long languageToken)
    {
        var (_, generation) = Create();
        var outputs = new List<Dictionary<string, ITensor>>();
        for (int request = 0; request < 2; request++)
        {
            int step = 0;
            var hidden = Hidden();
            var result = generation.Decode(hidden, WhisperTranscriptionOptions.ForLanguage(language), (first, feeds) =>
            {
                Assert.Equal(step == 0, first);
                if (first)
                {
                    Assert.Equal(2, feeds.Count);
                    Assert.Same(hidden, feeds["encoder_hidden_states"]);
                    Assert.Equal(new long[] { 50258, languageToken, 50360, 50364 }, ((Tensor<long>)feeds["input_ids"]).ToArray());
                }
                else
                {
                    Assert.Equal(17, feeds.Count);
                    Assert.Equal(new long[] { step == 1 ? 32 : 33 }, ((Tensor<long>)feeds["input_ids"]).ToArray());
                    foreach (var entry in feeds.Where(p => p.Key.StartsWith("past_key_values.")))
                    {
                        var origin = entry.Key.Contains(".encoder.") ? outputs[request * 3] : outputs[^1];
                        Assert.Same(origin["present." + entry.Key["past_key_values.".Length..]], entry.Value);
                    }
                }
                var next = Outputs(step, step == 0 ? 32 : step == 1 ? 33 : Eos, false, true);
                outputs.Add(next);step++;
                return next;
            }, CancellationToken.None);
            Assert.Equal(3, step);
            Assert.Equal("AB", result.Text);
            Assert.Equal(new[] { 32, 33, Eos }, result.TokenIds);
            Assert.Equal(WhisperStopReason.EndToken, result.StopReason);
            Assert.False(result.SkippedAsNoSpeech);
            Assert.Throws<NotSupportedException>(() => ((IList<int>)result.TokenIds)[0] = 3);
        }
    }

    [Fact]
    public void SuppressionAppliesToFirstEosGlobalTokensAndAllControlTokens()
    {
        var (_, generation) = Create();int step = 0;
        var result = generation.Decode(Hidden(), WhisperTranscriptionOptions.ForLanguage("en"), (first, _) =>
        {
            var output = Outputs(step, first ? 32 : Eos, false, true);
            var logits = ((DenseTensor<float>)output["logits"]).Buffer.Span;
            int offset = logits.Length - Vocab;
            foreach (int id in new[] { 1, 50364, 50365, 51865, 50259 }) logits[offset + id] = 100;
            if (first) logits[offset + Eos] = 100;
            step++;return output;
        }, CancellationToken.None);
        Assert.Equal(new[] { 32, Eos }, result.TokenIds);
    }

    [Theory]
    [InlineData(true, false, true)]
    [InlineData(true, true, false)]
    [InlineData(false, false, false)]
    public void NoSpeechUsesFirstPositionAndConfidenceOverride(bool noSpeech, bool confident, bool skipped)
    {
        var (_, generation) = Create();int step = 0;
        var result = generation.Decode(Hidden(), WhisperTranscriptionOptions.ForLanguage("en"), (_, _) =>
            Outputs(step, step++ == 0 ? 32 : Eos, noSpeech, confident), CancellationToken.None);
        Assert.Equal(skipped, result.SkippedAsNoSpeech);
        Assert.Equal(skipped ? "" : "A", result.Text);
        Assert.Equal(noSpeech, result.NoSpeechProbability > 0.6);
        Assert.Equal(confident, result.AverageLogProbability > -1);
    }

    [Fact]
    public void NoSpeechPolicyCanBeDisabledAndLogProbabilityOverrideCanBeDisabled()
    {
        var (_, generation) = Create();
        foreach (bool enabled in new[] { false, true })
        {
            int step = 0;
            var options = new WhisperTranscriptionOptions("en", 4, enabled ? 0.6 : null, null);
            var result = generation.Decode(Hidden(), options, (_, _) => Outputs(step, step++ == 0 ? 32 : Eos, true, true), CancellationToken.None);
            Assert.Equal(enabled, result.SkippedAsNoSpeech);
        }
    }

    [Fact]
    public void TokenLimitIsExplicitAndDoesNotFabricateEos()
    {
        var (_, generation) = Create();int step = 0;
        var options = new WhisperTranscriptionOptions("en", 2, null, null);
        var result = generation.Decode(Hidden(), options, (_, _) => Outputs(step++, 32, false, true), CancellationToken.None);
        Assert.Equal(2, step);
        Assert.Equal("AA", result.Text);
        Assert.Equal(new[] { 32, 32 }, result.TokenIds);
        Assert.Equal(WhisperStopReason.TokenLimit, result.StopReason);
    }

    [Theory]
    [InlineData(0)] [InlineData(445)]
    public void InvalidLimitsAreRejectedBeforeInference(int limit)
    {
        var (_, generation) = Create();
        Assert.Throws<ArgumentOutOfRangeException>(() => generation.Decode(Hidden(), new WhisperTranscriptionOptions("en", limit, null, null),
            (_, _) => throw new Exception("Inference must not run"), CancellationToken.None));
    }

    [Fact]
    public void InvalidLanguageAndThresholdsAreRejected()
    {
        var (_, generation) = Create();
        Assert.Throws<ArgumentException>(() => generation.Validate(WhisperTranscriptionOptions.ForLanguage("unsupported")));
        foreach (double value in new[] { double.NaN, double.PositiveInfinity, -0.1, 1.1 })
            Assert.Throws<ArgumentOutOfRangeException>(() => generation.Validate(new WhisperTranscriptionOptions("en", 1, value, null)));
        foreach (double value in new[] { double.NaN, double.NegativeInfinity, 0.1 })
            Assert.Throws<ArgumentOutOfRangeException>(() => generation.Validate(new WhisperTranscriptionOptions("en", 1, null, value)));
    }

    [Theory]
    [InlineData("nonfinite")] [InlineData("logit-shape")] [InlineData("missing-cache")] [InlineData("cache-shape")]
    public void InvalidOutputsFailAndNextRequestStartsFresh(string defect)
    {
        var (_, generation) = Create();
        Assert.Throws<InvalidDataException>(() => generation.Decode(Hidden(), WhisperTranscriptionOptions.ForLanguage("en"), (_, _) =>
        {
            var output = Outputs(0, 32, false, true);
            if (defect == "nonfinite") ((DenseTensor<float>)output["logits"]).Buffer.Span[0] = float.NaN;
            if (defect == "logit-shape") output["logits"] = new DenseTensor<float>(new[] { 1, 1, Vocab });
            if (defect == "missing-cache") output.Remove("present.0.decoder.key");
            if (defect == "cache-shape") output["present.0.encoder.key"] = new DenseTensor<float>(new[] { 1, 20, 1, 64 });
            return output;
        }, CancellationToken.None));
        int count = 0;
        var result = generation.Decode(Hidden(), new WhisperTranscriptionOptions("en", 1, null, null), (first, feeds) =>
        {
            Assert.True(first);Assert.Equal(2, feeds.Count);count++;return Outputs(0, 32, false, true);
        }, CancellationToken.None);
        Assert.Equal(1, count);Assert.Equal("A", result.Text);
    }

    [Fact]
    public void CancellationStopsBeforeTheNextModelCall()
    {
        var (_, generation) = Create();using var source = new CancellationTokenSource();int count = 0;
        Assert.Throws<OperationCanceledException>(() => generation.Decode(Hidden(), WhisperTranscriptionOptions.ForLanguage("en"), (_, _) =>
        {
            count++;source.Cancel();return Outputs(0, 32, false, true);
        }, source.Token));
        Assert.Equal(1, count);
        Assert.Throws<OperationCanceledException>(() => generation.Decode(Hidden(), WhisperTranscriptionOptions.ForLanguage("en"),
            (_, _) => throw new Exception("Inference must not run"), source.Token));
    }
}
