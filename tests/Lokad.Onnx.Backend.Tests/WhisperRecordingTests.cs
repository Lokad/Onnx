namespace Lokad.Onnx.Backend.Tests;

using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Text;
using System.Text.Json;

public class WhisperRecordingTests
{
    const int Begin = 50365, End = 50257, Rate = 16000;
    static Type Internal(string name) => typeof(WhisperTranscriber).Assembly.GetType("Lokad.Onnx." + name, true)!;
    static object Unwrap(Func<object> call)
    {
        try { return call(); }
        catch (TargetInvocationException e) when (e.InnerException is not null)
        { ExceptionDispatchInfo.Capture(e.InnerException).Throw(); throw; }
    }
    delegate void ApplyRules(Span<float> logits, IReadOnlyList<int> tokens, int end);

    [Fact]
    public void TimestampMasksAndChoicesMatchPinnedOriginalWhisper()
    {
        var apply = Internal("WhisperTimestampRules").GetMethod("Apply", BindingFlags.Static | BindingFlags.NonPublic)!.CreateDelegate<ApplyRules>();
        using var fixture = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "whisper-timestamps.json")));
        Assert.Equal(50, fixture.RootElement.GetProperty("cases").GetArrayLength());
        foreach (var row in fixture.RootElement.GetProperty("cases").EnumerateArray())
        {
            var logits = Enumerable.Repeat(row.GetProperty("baseline").GetSingle(), 51866).ToArray();
            foreach (var item in row.GetProperty("overrides").EnumerateObject()) logits[int.Parse(item.Name)] = item.Value.GetSingle();
            Array.Fill(logits, float.NegativeInfinity, End + 1, Begin - End - 1);
            apply(logits, row.GetProperty("tokens").EnumerateArray().Select(t => t.GetInt32()).ToArray(), End);
            var allowed = new bool[logits.Length];
            foreach (var range in row.GetProperty("allowed").EnumerateArray())
                Array.Fill(allowed, true, range[0].GetInt32(), range[1].GetInt32() - range[0].GetInt32());
            Assert.Equal(allowed, logits.Select(float.IsFinite).ToArray());
            int best = 0;
            for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
            Assert.Equal(row.GetProperty("argmax").GetInt32(), best);
        }
    }

    static string Text(IEnumerable<int> tokens) => string.Concat(tokens.Select(t => (char)('A' + t - 32)));
    static WhisperTranscription Result(int[] tokens) => Result(tokens, WhisperStopReason.EndToken, false);
    static WhisperTranscription Result(int[] tokens, bool skipped) => Result(tokens, WhisperStopReason.EndToken, skipped);
    static WhisperTranscription Result(int[] tokens, WhisperStopReason stop) => Result(tokens, stop, false);
    static WhisperTranscription Result(int[] tokens, WhisperStopReason stop, bool skipped)
        => new(skipped ? "" : Text(tokens.Where(t => t < End)), Array.AsReadOnly(tokens), stop, skipped, .01, -.1);
    static WhisperRecording Run(int samples, Func<int, int, WhisperTranscription> decode) => Run(samples, decode, 256, CancellationToken.None);
    static WhisperRecording Run(int samples, Func<int, int, WhisperTranscription> decode, int windows) => Run(samples, decode, windows, CancellationToken.None);
    static WhisperRecording Run(int samples, Func<int, int, WhisperTranscription> decode, CancellationToken cancellation) => Run(samples, decode, 256, cancellation);
    static WhisperRecording Run(int samples, Func<int, int, WhisperTranscription> decode, int windows, CancellationToken cancellation)
        => (WhisperRecording)Unwrap(() => Internal("WhisperRecordingPolicy").GetMethod("Run", BindingFlags.Static | BindingFlags.NonPublic)!
            .Invoke(null, new object[] { samples, windows, decode, (Func<IEnumerable<int>, string>)Text, cancellation })!);

    [Fact]
    public void UnfinishedBoundaryIsDecodedAgainAndOnlyCompletedTextIsCommitted()
    {
        var calls = new List<(int, int)>();
        var result = Run(35 * Rate, (start, length) =>
        {
            calls.Add((start, length));
            return calls.Count == 1 ? Result(new[] { Begin,32,Begin+1000,Begin+1000,34,End })
                : Result(new[] { Begin,33,Begin+500,End });
        });
        Assert.Equal(new[] { (0,30*Rate),(20*Rate,15*Rate) }, calls);
        Assert.Equal("AB", result.Text);
        Assert.Equal("AC", result.Windows[0].Decoding.Text);
        Assert.Equal(WhisperRecordingStopReason.Completed, result.StopReason);
        Assert.Equal(35, result.ProcessedSeconds);
        Assert.Equal(new[] { (0d,20d),(20d,30d) }, result.Segments.Select(s => (s.StartSeconds,s.EndSeconds)));
        Assert.Throws<NotSupportedException>(() => ((IList<int>)result.Segments[0].TokenIds)[0] = 42);
        Assert.Throws<NotSupportedException>(() => ((IList<WhisperWindow>)result.Windows).Clear());
        Assert.Throws<NotSupportedException>(() => ((IList<WhisperSegment>)result.Segments).Clear());
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void FinalTimestampAdvancesWholeWindowAndClipsPublicIntervals(bool multiple)
    {
        int[] tokens = multiple ? new[] { Begin,32,Begin+250,Begin+250,33,Begin+1500,End }
            : new[] { Begin+50,32,Begin+1500,End };
        var result = Run(10*Rate+1, (_, _) => Result(tokens));
        Assert.Equal(WhisperRecordingStopReason.Completed, result.StopReason);
        Assert.Equal(10+1d/Rate, result.ProcessedSeconds);
        Assert.Equal(10+1d/Rate, result.Segments[^1].EndSeconds);
        Assert.Equal(multiple ? "AB" : "A", result.Text);
    }

    [Theory]
    [InlineData("open", "", 0)]
    [InlineData("closed", "A", 10)]
    [InlineData("tail", "A", 10)]
    public void TokenLimitPreservesRawTailWithoutClaimingItProcessed(string kind, string expected, int seconds)
    {
        int[] tokens = kind == "open" ? new[] { Begin,32 } : kind == "closed" ? new[] { Begin,32,Begin+500 }
            : new[] { Begin,32,Begin+500,Begin+500,33 };
        var result = Run(60*Rate, (_, _) => Result(tokens, WhisperStopReason.TokenLimit));
        Assert.Equal(WhisperRecordingStopReason.TokenLimit, result.StopReason);
        Assert.Equal(expected, result.Text); Assert.Equal(seconds, result.ProcessedSeconds);
        Assert.Single(result.Windows); Assert.Equal(tokens, result.Windows[0].Decoding.TokenIds);
    }

    [Fact]
    public void WindowLimitReportsCommittedPositionAndCompletedAtExactBoundaryWins()
    {
        WhisperTranscription Decode(int _, int __) => Result(new[] { Begin,32,Begin+1500,End });
        var incomplete = Run(60*Rate+1, Decode, 2);
        Assert.Equal(WhisperRecordingStopReason.WindowLimit, incomplete.StopReason);
        Assert.Equal(60, incomplete.ProcessedSeconds); Assert.Equal(2,incomplete.Windows.Count);
        Assert.Equal(WhisperRecordingStopReason.Completed, Run(60*Rate, Decode, 2).StopReason);
    }

    [Fact]
    public void EmptySilentAndNoSpeechWindowsAreExplicit()
    {
        Assert.Empty(Run(0, (_, _) => throw new Exception("No window expected")).Windows);
        var silent = Run(600*Rate, (_, _) => Result(Array.Empty<int>(), WhisperStopReason.SilentInput, true));
        Assert.Equal(20,silent.Windows.Count); Assert.Empty(silent.Text); Assert.Equal(600,silent.ProcessedSeconds);
        var skipped = Run(40*Rate, (_, _) => Result(new[] { Begin,32,Begin+500,End }, skipped:true));
        Assert.Empty(skipped.Segments); Assert.Equal(40,skipped.ProcessedSeconds);
        var limited = Run(Rate, (_, _) => Result(new[] { Begin,32 }, WhisperStopReason.TokenLimit, true));
        Assert.Equal(WhisperRecordingStopReason.TokenLimit,limited.StopReason); Assert.Equal(0,limited.ProcessedSeconds);
    }

    [Fact]
    public void ZeroAdvanceAndMalformedTokensDoNotLoop()
    {
        var result = Run(60*Rate, (_, _) => Result(new[] { Begin,Begin,End }));
        Assert.Single(result.Windows); Assert.Equal(0,result.ProcessedSeconds);
        Assert.Equal(WhisperRecordingStopReason.NoProgress,result.StopReason);
        Assert.Throws<InvalidDataException>(() => Run(Rate, (_, _) => Result(new[] { Begin,50360,End })));
        Assert.Throws<InvalidDataException>(() => Run(Rate, (_, _) => Result(new[] { Begin,32,End,33,End })));
        Assert.Throws<InvalidDataException>(() => Run(60*Rate, (_, _) => Result(new[] { Begin+500,32,Begin+250,Begin+600,End })));
    }

    [Fact]
    public void CancellationBeforeAndAfterWindowDoesNotPublishPartialResultsAndNextCallWorks()
    {
        using var canceled = new CancellationTokenSource(); canceled.Cancel();
        Assert.Throws<OperationCanceledException>(() => Run(Rate, (_, _) => throw new Exception("No call expected"), cancellation:canceled.Token));
        using var during = new CancellationTokenSource();
        Assert.Throws<OperationCanceledException>(() => Run(Rate, (_, _) => { during.Cancel();return Result(new[] { Begin,32,Begin+10,End }); }, cancellation:during.Token));
        Assert.Equal("A",Run(Rate, (_, _) => Result(new[] { Begin,32,Begin+10,End })).Text);
    }

    [Fact]
    public void TimestampGenerationUsesThreeTokenPrefixAndIndependentCacheLengths()
    {
        using var fixture=JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,"fixtures","whisper-text.json")));
        using var tokenizerFile=new MemoryStream(Encoding.UTF8.GetBytes(fixture.RootElement.GetProperty("tokenizer").GetRawText()));
        using var configFile=new MemoryStream(Encoding.UTF8.GetBytes(fixture.RootElement.GetProperty("generation").GetRawText()));
        object tokenizer=Activator.CreateInstance(Internal("WhisperTokenizer"),BindingFlags.Instance|BindingFlags.NonPublic,null,new object[]{tokenizerFile},null)!;
        object generation=Activator.CreateInstance(Internal("WhisperGeneration"),BindingFlags.Instance|BindingFlags.NonPublic,null,new[]{configFile,tokenizer},null)!;
        var decode=Internal("WhisperGeneration").GetMethod("DecodeTimestamps",BindingFlags.Instance|BindingFlags.NonPublic)!;
        int[] sequence={Begin,32,Begin+50,End};
        for(int request=0;request<2;request++)
        {
            int step=0;
            var previous=new List<Dictionary<string,ITensor>>();
            Func<bool,Dictionary<string,ITensor>,IReadOnlyDictionary<string,ITensor>> execute=(first,feeds)=>
            {
                Assert.Equal(step==0,first);
                Assert.Equal(first?new long[]{50258,50259,50360}:new long[]{sequence[step-1]},((Tensor<long>)feeds["input_ids"]).ToArray());
                if(!first) foreach(var item in feeds.Where(p=>p.Key.StartsWith("past_key_values.")))
                    Assert.Same(previous[item.Key.Contains(".encoder.")?0:step-1]["present."+item.Key["past_key_values.".Length..]],item.Value);
                int positions=first?3:1;
                var scores=Enumerable.Repeat(-20f,positions*51866).ToArray();
                scores[(positions-1)*51866+sequence[step]]=20;
                var output=new Dictionary<string,ITensor>{{"logits",new DenseTensor<float>(scores,new[]{1,positions,51866})}};
                for(int layer=0;layer<4;layer++) foreach(string kind in new[]{"key","value"})
                {
                    output.Add($"present.{layer}.decoder.{kind}",new DenseTensor<float>(new[]{1,20,3+step,64}));
                    if(first) output.Add($"present.{layer}.encoder.{kind}",new DenseTensor<float>(new[]{1,20,2,64}));
                }
                previous.Add(output);step++;return output;
            };
            var result=(WhisperTranscription)Unwrap(()=>decode.Invoke(generation,new object[]{new DenseTensor<float>(new[]{1,2,1280}),WhisperTranscriptionOptions.ForLanguage("en"),execute,CancellationToken.None})!);
            Assert.Equal(sequence,result.TokenIds);Assert.Equal("A",result.Text);Assert.Equal(4,step);
        }
    }
}
