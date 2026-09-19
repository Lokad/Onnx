namespace Lokad.Onnx.Backend.Tests;

using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Text;
using System.Text.Json;

public class ParakeetGenerationTests
{
    const int Blank = 8192;
    static object Call(Func<object?> call)
    {
        try { return call() ?? throw new InvalidOperationException("Null reflection result"); }
        catch (TargetInvocationException e) when (e.InnerException is not null)
        { ExceptionDispatchInfo.Capture(e.InnerException).Throw(); throw; }
    }
    static Type Type(string name) => typeof(ParakeetTranscriber).Assembly.GetType("Lokad.Onnx." + name, true)
        ?? throw new InvalidOperationException("Missing internal policy");
    static object Create(string name, params object[] args) => Call(() => Activator.CreateInstance(Type(name),
        BindingFlags.Instance | BindingFlags.NonPublic, null, args, null));
    static object Invoke(object target, string name, params object[] args) => Call(() =>
        (target.GetType().GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("Missing policy method")).Invoke(target, args));
    static string Vocabulary()
    {
        var text = new StringBuilder();
        for (int i = 0; i <= Blank; i++) text.Append(i == Blank ? "<blk>" : "\u2581p" + i).Append(' ').Append(i).Append('\n');
        return text.ToString();
    }
    static object Vocab(string text)
    {
        using var input = new MemoryStream(Encoding.UTF8.GetBytes(text));
        return Create("ParakeetVocabulary", input);
    }
    static Tensor<float> Hidden(int frames) => new DenseTensor<float>(Enumerable.Range(0, 1024 * frames).Select(i => (float)i).ToArray(), new[] { 1, 1024, frames });
    static Dictionary<string, ITensor> Output(int token, int duration, float state1, float state2)
    {
        var logits = Enumerable.Repeat(-10f, 8198).ToArray(); logits[token] = 10; logits[8193 + duration] = 10;
        return new Dictionary<string, ITensor>
        {
            ["outputs"] = new DenseTensor<float>(logits, new[] { 1, 1, 1, 8198 }),
            ["prednet_lengths"] = new DenseTensor<int>(new[] { 1 }, new[] { 1 }),
            ["output_states_1"] = new DenseTensor<float>(Enumerable.Repeat(state1, 1280).ToArray(), new[] { 2, 1, 640 }),
            ["output_states_2"] = new DenseTensor<float>(Enumerable.Repeat(state2, 1280).ToArray(), new[] { 2, 1, 640 })
        };
    }
    static ParakeetTranscription Decode(Tensor<float> hidden, int frames, ParakeetTranscriptionOptions options,
        Func<Dictionary<string, ITensor>, IReadOnlyDictionary<string, ITensor>> execute, CancellationToken cancellation)
    {
        object generation = Create("ParakeetGeneration", Vocab(Vocabulary()));
        return (ParakeetTranscription)Invoke(generation, "Decode", hidden, frames, options, execute, cancellation);
    }

    [Theory]
    [InlineData(1)] [InlineData(2)] [InlineData(3)] [InlineData(4)]
    public void PositiveDurationAtSymbolLimitAdvancesExactlyOnce(int duration)
    {
        int calls = 0;
        var result = Decode(Hidden(10), 10, new(4096, 1), feeds =>
        {
            var frame = ((Tensor<float>)feeds["encoder_outputs"]).ToArray();
            Assert.Equal(calls * duration, frame[0]);
            Assert.Equal(10 * 1023 + calls * duration, frame[1023]);
            calls++; return Output(7, duration, calls, -calls);
        }, CancellationToken.None);
        Assert.Equal(Enumerable.Range(0, (10 + duration - 1) / duration).Select(i => i * duration), result.FrameIndices);
        Assert.All(result.TokenIds, v => Assert.Equal(7, v));
        Assert.All(result.DurationFrames, v => Assert.Equal(duration, v));
        Assert.Equal(ParakeetStopReason.EndOfAudio, result.StopReason);
    }

    [Fact]
    public void BlankRollsBackBothStatesAndKeepsLastToken()
    {
        int calls = 0;
        var observed = new List<(int Target, float First, float Second, float Frame)>();
        var result = Decode(Hidden(4), 4, new(10, 10), feeds =>
        {
            observed.Add((((Tensor<int>)feeds["targets"]).ToArray()[0],
                ((Tensor<float>)feeds["input_states_1"]).ToArray()[0], ((Tensor<float>)feeds["input_states_2"]).ToArray()[0],
                ((Tensor<float>)feeds["encoder_outputs"]).ToArray()[0]));
            calls++;
            return calls switch { 1 => Output(Blank, 0, 10, 20), 2 => Output(5, 0, 30, 40),
                3 => Output(Blank, 2, 50, 60), _ => Output(5, 1, 70, 80) };
        }, CancellationToken.None);
        Assert.Equal(new[] { (Blank, 0f, 0f, 0f), (Blank, 0f, 0f, 1f), (5, 30f, 40f, 1f), (5, 30f, 40f, 3f) }, observed);
        Assert.Equal(new[] { 5, 5 }, result.TokenIds);
        Assert.Equal(new[] { 1, 3 }, result.FrameIndices);
        Assert.Equal(new[] { 0, 1 }, result.DurationFrames);
        Assert.Equal("p5 p5", result.Text);
        Assert.Equal(4, result.DecoderCalls);
    }

    [Theory]
    [InlineData(1)] [InlineData(2)] [InlineData(10)]
    public void ZeroDurationEmissionCapGuaranteesProgress(int maximum)
    {
        var result = Decode(Hidden(3), 3, new(4096, maximum), _ => Output(2, 0, 1, 2), CancellationToken.None);
        Assert.Equal(3 * maximum, result.DecoderCalls);
        Assert.Equal(Enumerable.Range(0, 3).SelectMany(i => Enumerable.Repeat(i, maximum)), result.FrameIndices);
    }

    [Fact]
    public void TokenLimitPreservesPartialDecisionsAndHeldResult()
    {
        Tensor<float> hidden = Hidden(3); float[] before = hidden.ToArray();
        var result = Decode(hidden, 3, new(2, 10), _ => Output(1, 0, 3, 4), CancellationToken.None);
        Assert.Equal(ParakeetStopReason.TokenLimit, result.StopReason);
        Assert.Equal(new[] { 1, 1 }, result.TokenIds);
        Assert.Equal(new[] { 0, 0 }, result.FrameIndices);
        Assert.Equal(2, result.DecoderCalls);
        Assert.Throws<NotSupportedException>(() => ((IList<int>)result.TokenIds)[0] = 2);
        Assert.Throws<NotSupportedException>(() => ((IList<int>)result.FrameIndices)[0] = 2);
        Assert.Throws<NotSupportedException>(() => ((IList<int>)result.DurationFrames)[0] = 2);
        Decode(hidden, 3, new(3, 10), _ => Output(2, 1, 7, 8), CancellationToken.None);
        Assert.Equal(before, hidden.ToArray());
        Assert.Equal(new[] { 1, 1 }, result.TokenIds);
    }

    [Fact]
    public void ExhaustionAtTokenLimitIsNaturalCompletion()
    {
        var result = Decode(Hidden(1), 1, new(1, 10), _ => Output(1, 4, 1, 2), CancellationToken.None);
        Assert.Equal(ParakeetStopReason.EndOfAudio, result.StopReason);
    }

    [Fact]
    public void ValidLengthStopsBeforePaddedFrames()
    {
        var result = Decode(Hidden(8), 2, new(100, 10), _ => Output(Blank, 1, 1, 2), CancellationToken.None);
        Assert.Equal(2, result.DecoderCalls); Assert.Empty(result.TokenIds); Assert.Equal("", result.Text);
        var empty = Decode(Hidden(8), 0, new(100, 10), _ => throw new Exception("No call allowed"), CancellationToken.None);
        Assert.Equal(0, empty.DecoderCalls);
    }

    [Fact]
    public void ArgmaxTiesChooseFirstIndex()
    {
        var result = Decode(Hidden(1), 1, new(1, 1), _ =>
        {
            var output = Output(0, 0, 1, 2);
            output["outputs"] = new DenseTensor<float>(new float[8198], new[] { 1, 1, 1, 8198 });
            return output;
        }, CancellationToken.None);
        Assert.Equal(new[] { 0 }, result.TokenIds); Assert.Equal(new[] { 0 }, result.DurationFrames);
    }

    [Theory]
    [InlineData(0, 10)] [InlineData(4097, 10)] [InlineData(10, 0)] [InlineData(10, 11)]
    public void LimitsAreValidatedBeforeAnyCall(int tokens, int perFrame) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => Decode(Hidden(1), 1, new(tokens, perFrame),
            _ => throw new Exception("Unexpected call"), CancellationToken.None));

    [Fact]
    public void CancellationBeforeAndAfterGraphCall()
    {
        using var canceled = new CancellationTokenSource(); canceled.Cancel();
        Assert.Throws<OperationCanceledException>(() => Decode(Hidden(1), 1, new(10, 10),
            _ => throw new Exception("Unexpected call"), canceled.Token));
        using var mid = new CancellationTokenSource();
        Assert.Throws<OperationCanceledException>(() => Decode(Hidden(1), 1, new(10, 10),
            _ => { mid.Cancel(); return Output(1, 1, 1, 2); }, mid.Token));
    }

    [Theory]
    [InlineData("logit-nan")] [InlineData("logit-inf")] [InlineData("logit-shape")]
    [InlineData("state1")] [InlineData("state2")] [InlineData("length-value")] [InlineData("length-type")]
    [InlineData("missing")]
    public void MalformedDecoderResultsAreRejected(string failure)
    {
        Assert.Throws<InvalidDataException>(() => Decode(Hidden(1), 1, new(10, 10), _ =>
        {
            var output = Output(Blank, 1, 1, 2);
            switch (failure)
            {
                case "logit-nan": case "logit-inf":
                    var logits = ((Tensor<float>)output["outputs"]).ToArray(); logits[8197] = failure == "logit-nan" ? float.NaN : float.PositiveInfinity;
                    output["outputs"] = new DenseTensor<float>(logits, new[] { 1, 1, 1, 8198 }); break;
                case "logit-shape": output["outputs"] = new DenseTensor<float>(new[] { 8198 }); break;
                case "state1": output["output_states_1"] = new DenseTensor<float>(new[] { 1, 1, 1280 }); break;
                case "state2": output["output_states_2"] = new DenseTensor<float>(Enumerable.Repeat(float.NaN, 1280).ToArray(), new[] { 2, 1, 640 }); break;
                case "length-value": output["prednet_lengths"] = new DenseTensor<int>(new[] { 2 }, new[] { 1 }); break;
                case "length-type": output["prednet_lengths"] = new DenseTensor<long>(new[] { 1L }, new[] { 1 }); break;
                default: output.Remove("output_states_1"); break;
            }
            return output;
        }, CancellationToken.None));
        Assert.Equal(1, Decode(Hidden(1), 1, new(10, 10), _ => Output(1, 1, 1, 2), CancellationToken.None).DecoderCalls);
    }

    [Theory]
    [InlineData(-1)] [InlineData(3)]
    public void InvalidEncoderLengthRejected(int length) => Assert.Throws<InvalidDataException>(() =>
        Decode(Hidden(2), length, new(10, 10), _ => Output(1, 1, 1, 2), CancellationToken.None));

    [Theory]
    [InlineData("missing")] [InlineData("id")] [InlineData("duplicate")]
    [InlineData("blank")] [InlineData("whitespace")] [InlineData("extra")]
    public void InvalidVocabularyRejected(string kind)
    {
        string text = Vocabulary();
        text = kind switch { "missing" => text.Replace("\u2581p1 1\n", ""), "id" => text.Replace("\u2581p1 1\n", "\u2581p1 2\n"),
            "duplicate" => text.Replace("\u2581p1 1\n", "\u2581p0 1\n"), "blank" => text.Replace("<blk>", "wrong"),
            "whitespace" => text.Replace("\u2581p1", "bad piece"), _ => text + "extra 8193\n" };
        Assert.Throws<InvalidDataException>(() => Vocab(text));
    }

    [Fact]
    public void InvalidUtf8Rejected()
    {
        using var bytes = new MemoryStream(new byte[] { 0xC0, 0xAF });
        Assert.Throws<DecoderFallbackException>(() => Create("ParakeetVocabulary", bytes));
    }

    [Fact]
    public void VocabularyUnicodePunctuationAndWhitespaceMatchReferenceRules()
    {
        string text = Vocabulary().Replace("\u2581p0", "\u2581Bonjour").Replace("\u2581p1 1\n", "\u2581, 1\n")
            .Replace("\u2581p2 2\n", "\u2581été 2\n").Replace("\u2581p3 3\n", "\u2581𐐀 3\n")
            .Replace("\u2581p4 4\n", "\u2581! 4\n").Replace("\u2581p5 5\n", "\u2581 5\n");
        object vocabulary = Vocab(text);
        Assert.Equal("Bonjour, été 𐐀!", (string)Invoke(vocabulary, "Decode", new[] { 0, 1, 2, 3, 4, 5 }));
        Assert.Equal(" Bonjour", (string)Invoke(vocabulary, "Decode", new[] { 5, 0 }));
        Assert.Throws<InvalidDataException>(() => Invoke(vocabulary, "Decode", new[] { Blank }));
    }

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)]
    [InlineData(4)] [InlineData(5)] [InlineData(6)] [InlineData(7)]
    [InlineData(8)] [InlineData(9)] [InlineData(10)] [InlineData(11)]
    public void DecodeMatchesIndependentPythonUnicodeFixture(int index)
    {
        using var fixture = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "parakeet-text.json")));
        var pieces = fixture.RootElement.GetProperty("pieces").EnumerateArray().Select(p => p.GetString()).ToArray();
        var text = new StringBuilder();
        for (int i = 0; i <= Blank; i++) text.Append(i < pieces.Length ? pieces[i] : i == Blank ? "<blk>" : "\u2581p" + i).Append(' ').Append(i).Append('\n');
        object vocabulary = Vocab(text.ToString());
        var row = fixture.RootElement.GetProperty("cases")[index];
        Assert.Equal(row.GetProperty("text").GetString(), (string)Invoke(vocabulary, "Decode", row.GetProperty("tokens").EnumerateArray().Select(v => v.GetInt32()).ToArray()));
    }
}
