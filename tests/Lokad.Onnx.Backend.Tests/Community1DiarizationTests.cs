namespace Lokad.Onnx.Backend.Tests;

using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Text.Json;
using Google.Protobuf;
using global::Onnx;

public class Community1DiarizationTests
{
    static object Call(string type, string method, params object[] args)
    {
        var t = typeof(Community1Diarizer).Assembly.GetType("Lokad.Onnx." + type, true) ?? throw new InvalidOperationException();
        try { return t.GetMethod(method, BindingFlags.NonPublic | BindingFlags.Static)?.Invoke(null, args) ?? throw new InvalidOperationException(); }
        catch (TargetInvocationException e) when (e.InnerException is not null) { ExceptionDispatchInfo.Capture(e.InnerException).Throw(); throw; }
    }
    static T Property<T>(object value, string name) => (T)(value.GetType().GetProperty(name)?.GetValue(value) ?? throw new InvalidOperationException());
    static int[] Count(bool[] a, int chunks) => (int[])Call("Community1Timeline", "Count", a, chunks, CancellationToken.None);
    static bool[] Activity(int chunks, Func<int, int, int, bool> value) => Enumerable.Range(0, chunks * 589 * 3).Select(i => value(i / (589 * 3), i / 3 % 589, i % 3)).ToArray();

    [Theory]
    [InlineData(1, 1)] [InlineData(159999, 1)] [InlineData(160000, 1)] [InlineData(160001, 2)]
    [InlineData(176000, 2)] [InlineData(176001, 3)] [InlineData(9600000, 591)]
    public void ChunkBoundariesKeepExactlyOnePaddedTail(int samples, int chunks) =>
        Assert.Equal(chunks, (int)Call("Community1Timeline", "ChunkCount", samples));

    [Fact]
    public void WindowsPreserveOffsetsAndOwnTheirPadding()
    {
        var input = Enumerable.Range(0, 160001).Select(i => (float)i).ToArray();
        var window = (float[])Call("Community1Timeline", "Window", input, 1);
        Assert.Equal(160000, window.Length); Assert.Equal(16000, window[0]); Assert.Equal(160000, window[144000]); Assert.All(window.Skip(144001), v => Assert.Equal(0, v));
        window[0] = -1; Assert.Equal(16000, input[16000]);
        Assert.Throws<ArgumentOutOfRangeException>(() => Call("Community1Timeline", "Window", input, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => Call("Community1Timeline", "ChunkCount", 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Call("Community1Timeline", "ChunkCount", 9600001));
    }

    [Fact]
    public void PowersetMapsAllClassesAndRejectsNonfiniteDiscardedScores()
    {
        var scores = new float[589 * 7];
        for (int t = 0; t < 7; t++) scores[t * 7 + t] = 1;
        var result = (bool[])Call("Community1Timeline", "Powerset", (object)scores);
        int[] bits = { 0, 1, 2, 4, 3, 5, 6 };
        for (int t = 0; t < 7; t++) for (int s = 0; s < 3; s++) Assert.Equal((bits[t] & (1 << s)) != 0, result[t * 3 + s]);
        Assert.All(result.Skip(21), Assert.False);
        scores[^1] = float.NaN; Assert.Throws<ArgumentException>(() => Call("Community1Timeline", "Powerset", (object)scores));
    }

    [Theory]
    [InlineData(2, 32)] [InlineData(3, 3)]
    public void CleanMaskThresholdIsStrictAndFallsBackToFullSpeech(int clean, int expected)
    {
        var a = Activity(1, (_, t, s) => s == 0 && t < clean || s < 2 && t >= 120 && t < 150);
        var masks = (float[][])Call("Community1Timeline", "EmbeddingMasks", (object)a);
        Assert.Equal(expected, masks[0].Sum()); Assert.Equal(30, masks[1].Sum()); Assert.All(masks[2], v => Assert.Equal(0, v));
        Assert.Equal(a, Activity(1, (_, t, s) => s == 0 && t < clean || s < 2 && t >= 120 && t < 150));
    }

    [Fact]
    public void OverlapCountsRoundHalfToEvenAndLeaveUncoveredTailEmpty()
    {
        var a = Activity(2, (c, _, s) => c == 0 && s == 0); var count = Count(a, 2);
        Assert.Equal(653, count.Length); Assert.All(count.Take(59), v => Assert.Equal(1, v));
        Assert.All(count.Skip(59), v => Assert.Equal(0, v)); // overlapping .5 rounds to zero.
        a = Activity(2, (c, _, s) => s == 0 || c == 1 && s == 1); count = Count(a, 2);
        Assert.Equal(2, count[59]); Assert.Equal(2, count[588]); Assert.Equal(2, count[647]); Assert.Equal(0, count[648]);
    }

    [Fact]
    public void MergedLocalSpeakersUseMaximumAndCountCanCreateAZeroCentroidSpeaker()
    {
        var a = Activity(1, (_, _, s) => s < 2); var count = Count(a, 1);
        var ordinary = Call("Community1Timeline", "Reconstruct", a, 1, new[] { 0, 0, -2 }, count, false, CancellationToken.None);
        Assert.Equal(2, Property<int>(ordinary, "Speakers")); var values = Property<bool[]>(ordinary, "Values");
        Assert.True(values[0]); Assert.True(values[1]);
        var exclusive = Call("Community1Timeline", "Reconstruct", a, 1, new[] { 0, 0, -2 }, count, true, CancellationToken.None);
        Assert.Equal(1, Property<int>(exclusive, "Speakers")); Assert.True(Property<bool[]>(exclusive, "Values")[0]);
    }

    [Fact]
    public void TiedVotesUseCanonicalLabelOrderAndFrameCentersDefineEndpoints()
    {
        var a = Activity(2, (c, _, s) => c == s); var count = Count(a, 2);
        var frames = Call("Community1Timeline", "Reconstruct", a, 2, new[] { 0, -2, -2, -2, 1, -2 }, count, true, CancellationToken.None);
        var values = Property<bool[]>(frames, "Values"); Assert.True(values[59 * 2]); Assert.False(values[59 * 2 + 1]);
        var intervals = (Array)Call("Community1Timeline", "Intervals", frames, CancellationToken.None);
        Assert.Equal(2, intervals.Length); var first = intervals.GetValue(0) ?? throw new InvalidOperationException();
        Assert.Equal(991.0 / 32000, Property<double>(first, "Start"));
        Assert.Equal(589 * 270.0 / 16000 + 991.0 / 32000, Property<double>(first, "End"), 12);
    }

    [Fact]
    public void SparsePipelineStatisticsRetainNativeEpsilonBehavior()
    {
        var tensor = new DenseTensor<float>(new float[] { 2, 9, 4, 7 }, new[] { 1, 2, 2 });
        var zero = (Tensor<float>)Call("WeSpeakerPooling", "PoolPipeline", tensor, new float[] { 0, 0 }, CancellationToken.None);
        Assert.Equal(new float[] { 0, 0, 0, 0 }, zero.ToArray());
        var one = (Tensor<float>)Call("WeSpeakerPooling", "PoolPipeline", tensor, new float[] { 1, 0 }, CancellationToken.None);
        Assert.Equal(new float[] { 2, 4, 0, 0 }, one.ToArray());
        Assert.Throws<ArgumentException>(() => Call("WeSpeakerPooling", "Pool", tensor, new float[] { 1, 0 }, true, CancellationToken.None));
    }

    [Fact]
    public void InvalidFramesCountsAndCancellationFailBeforeReconstruction()
    {
        var a = Activity(1, (_, _, _) => false); var counts = Count(a, 1);
        Assert.Throws<ArgumentException>(() => Call("Community1Timeline", "Count", new bool[1], 1, CancellationToken.None));
        Assert.Throws<OperationCanceledException>(() => Call("Community1Timeline", "Count", a, 1, new CancellationToken(true)));
        counts[0] = 4; Assert.Throws<ArgumentException>(() => Call("Community1Timeline", "Reconstruct", a, 1, new[] { 0, 1, 2 }, counts, false, CancellationToken.None));
        counts[0] = 0; Assert.Throws<ArgumentException>(() => Call("Community1Timeline", "Reconstruct", a, 1, new[] { 0, -1, 2 }, counts, false, CancellationToken.None));
        var empty = Call("Community1Timeline", "Reconstruct", a, 1, new[] { -2, -2, -2 }, counts, false, CancellationToken.None);
        Assert.Empty(Property<bool[]>(empty, "Values"));
    }

    sealed class Models : IDisposable
    {
        readonly string directory = Path.Combine(Path.GetTempPath(), "diarization-" + Guid.NewGuid().ToString("N"));
        internal Models(int activeFrames, bool overlap)
        {
            Directory.CreateDirectory(directory);
            float[] scores = Enumerable.Range(0, 589 * 7).Select(i => i % 7 == (i / 7 < activeFrames ? overlap ? 4 : 1 : 0) ? 1f : 0).ToArray();
            Save("seg.onnx", "waveform", new[] { 1, 1, 160000 }, "scores", new[] { 1, 589, 7 }, scores);
            Save("enc.onnx", "fbank_features", new[] { 1, 998, 80 }, "/resnet/pool/Reshape_output_0", new[] { 1, 2560, 125 }, Enumerable.Repeat(1f, 2560 * 125).ToArray());
            Save("projection.onnx", "pooled", new[] { 1, 5120 }, "embedding", new[] { 1, 256 }, Enumerable.Repeat(1f, 256).ToArray());
            File.WriteAllText(Path.Combine(directory, "plda.json"), JsonSerializer.Serialize(new { schema = 1, input_dimensions = 256, output_dimensions = 128,
                mean1 = new double[256], mean2 = new double[128], mean = new double[128], phi = Enumerable.Repeat(1d, 128).ToArray(),
                lda = Enumerable.Range(0, 256).Select(i => Enumerable.Range(0, 128).Select(j => i == j ? 1d : 0).ToArray()).ToArray(),
                transform = Enumerable.Range(0, 128).Select(i => Enumerable.Range(0, 128).Select(j => i == j ? 1d : 0).ToArray()).ToArray() }));
        }
        internal Community1Diarizer Create() => new(Path.Combine(directory, "seg.onnx"), Path.Combine(directory, "enc.onnx"), Path.Combine(directory, "projection.onnx"), Path.Combine(directory, "plda.json"));
        void Save(string path, string input, int[] inputShape, string output, int[] outputShape, float[] values)
        {
            ValueInfoProto Info(string name, int[] dimensions)
            {
                var shape = new TensorShapeProto(); foreach (int size in dimensions) shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = size });
                return new ValueInfoProto { Name = name, Type = new TypeProto { TensorType = new TypeProto.Types.Tensor { ElemType = 1, Shape = shape } } };
            }
            var model = new ModelProto { IrVersion = 9, Graph = new GraphProto { Name = "fixture" }, OpsetImport = { new OperatorSetIdProto { Domain = "", Version = 17 } } };
            model.Graph.Input.Add(Info(input, inputShape)); model.Graph.Output.Add(Info(output, outputShape));
            var tensor = new TensorProto { Name = "constant", DataType = 1 }; tensor.Dims.Add(outputShape.Select(v => (long)v)); tensor.FloatData.Add(values); model.Graph.Initializer.Add(tensor);
            var node = new NodeProto { OpType = "Identity" }; node.Input.Add("constant"); node.Output.Add(output); model.Graph.Node.Add(node);
            File.WriteAllBytes(Path.Combine(directory, path), model.ToByteArray());
        }
        public void Dispose() => Directory.Delete(directory, true);
    }

    [Fact]
    public void PublicPipelineClipsPaddingOwnsResultsAndRecoversAfterRequestErrors()
    {
        using var models = new Models(589, false); var api = models.Create(); var pcm = new float[16000];
        var result = api.Diarize(pcm, 16000, CancellationToken.None); Assert.Equal(Community1DiarizationStatus.Completed, result.Status);
        var interval = Assert.Single(result.Intervals); Assert.Equal(991.0 / 32000, interval.Start); Assert.Equal(1, interval.End); Assert.Equal(0, interval.Speaker);
        Assert.Equal(result.Intervals, result.ExclusiveIntervals); var speaker = Assert.Single(result.Speakers); Assert.True(speaker.HasEmbedding); Assert.All(speaker.Centroid, v => Assert.Equal(1, v));
        Assert.Throws<NotSupportedException>(() => ((IList<double>)speaker.Centroid)[0] = 0);
        Assert.Throws<OperationCanceledException>(() => api.Diarize(pcm, 16000, new CancellationToken(true)));
        Assert.Throws<ArgumentOutOfRangeException>(() => api.Diarize(pcm, 8000, CancellationToken.None));
        Assert.Throws<ArgumentOutOfRangeException>(() => api.Diarize(new float[Community1Diarizer.MaximumSamples + 1], 16000, CancellationToken.None));
        Assert.Throws<ArgumentException>(() => api.Diarize(new float[] { float.NaN }, 16000, CancellationToken.None));
        Assert.Throws<ArgumentException>(() => api.Diarize(new float[] { 1.1f }, 16000, CancellationToken.None));
        Parallel.For(0, 3, _ => Assert.Equal(result.Intervals, api.Diarize(pcm, 16000, CancellationToken.None).Intervals));
        Assert.All(pcm, v => Assert.Equal(0, v)); Assert.Equal(1, speaker.Centroid[0]);
        Assert.Equal(Community1DiarizationStatus.NoSpeech, api.Diarize(Array.Empty<float>(), 16000, CancellationToken.None).Status);
    }

    [Theory]
    [InlineData(0, false, Community1DiarizationStatus.NoSpeech)]
    [InlineData(117, false, Community1DiarizationStatus.NoUsableEmbeddings)]
    [InlineData(589, true, Community1DiarizationStatus.NoUsableEmbeddings)]
    public void PublicNoDataStatesDistinguishActivityFromUsableTraining(int active, bool overlap, Community1DiarizationStatus status)
    {
        using var models = new Models(active, overlap); var api = models.Create(); var result = api.Diarize(new float[16000], 16000, CancellationToken.None);
        Assert.Equal(status, result.Status); Assert.Empty(result.Intervals); Assert.Empty(result.ExclusiveIntervals); Assert.Empty(result.Speakers);
    }
}
