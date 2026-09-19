namespace Lokad.Onnx.Backend.Tests;

using System.Reflection;
using System.Runtime.ExceptionServices;
using Google.Protobuf;
using global::Onnx;

public class WeSpeakerEmbeddingTests
{
    static Tensor<float> Pool(float[] values, int channels, float[] weights, bool weighted, CancellationToken cancellation)
    {
        var type = typeof(WeSpeakerEmbedder).Assembly.GetType("Lokad.Onnx.WeSpeakerPooling", true)
            ?? throw new InvalidOperationException();
        try
        {
            return (Tensor<float>)(type.GetMethod("Pool", BindingFlags.NonPublic | BindingFlags.Static)
                ?? throw new InvalidOperationException()).Invoke(null, new object[] {
                    new DenseTensor<float>(values, new[] { 1, channels, values.Length / channels }), weights, weighted, cancellation })!;
        }
        catch (TargetInvocationException e) when (e.InnerException is not null)
        { ExceptionDispatchInfo.Capture(e.InnerException).Throw(); throw; }
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void StatisticsMatchIndependentMeanAndSampleDeviation(bool weighted)
    {
        var input = new float[] { 1, 3, 5, -2, 4, 10 }; var weights = new float[] { 1, 1, 1 };
        var result = Pool(input, 2, weights, weighted, CancellationToken.None);
        Assert.Equal(new[] { 1, 4 }, result.Dimensions.ToArray());
        Assert.Equal(new float[] { 3, 4, 2, 6 }, result.ToArray());
        Assert.Equal(new float[] { 1, 3, 5, -2, 4, 10 }, input);
        Assert.Equal(new float[] { 1, 1, 1 }, weights);
        Pool(new float[] { 10, 20 }, 1, new float[] { 1, 1 }, true, CancellationToken.None);
        Assert.Equal(new float[] { 3, 4, 2, 6 }, result.ToArray());
    }

    [Fact]
    public void FractionalWeightsUseUnbiasedWeightedVariance()
    {
        // Normalized weights 1/6, 1/3, 1/2: mean=14/3; sum(w*(x-mean)^2)=53/6,
        // denominator=sum(w)-sum(w^2)/sum(w)=11/12, variance=106/11.
        var result = Pool(new float[] { 1, 3, 7 }, 1, new float[] { .25f, .5f, .75f }, true, CancellationToken.None).ToArray();
        Assert.InRange(Math.Abs(result[0] - 14.0 / 3), 0, 1e-6);
        Assert.InRange(Math.Abs(result[1] - Math.Sqrt(106.0 / 11)), 0, 1e-6);
    }

    [Fact]
    public void ZeroWeightsExcludeFramesAndDoNotChangeInputs()
    {
        var result = Pool(new float[] { 1, 1000, 5 }, 1, new float[] { 1, 0, 1 }, true, CancellationToken.None).ToArray();
        Assert.Equal(3, result[0]); Assert.InRange(Math.Abs(result[1] - Math.Sqrt(8)), 0, 1e-6);
    }

    [Theory]
    [InlineData(float.NaN)] [InlineData(float.PositiveInfinity)] [InlineData(float.NegativeInfinity)]
    public void NonfiniteBackboneEvenInExcludedFramesFails(float bad) => Assert.Throws<InvalidDataException>(() =>
        Pool(new float[] { 1, bad, 5 }, 1, new float[] { 1, 0, 1 }, true, CancellationToken.None));

    [Fact]
    public void PoolCancellationAndDegenerateMasksFailExplicitly()
    {
        Assert.Throws<OperationCanceledException>(() => Pool(new float[] { 1, 3 }, 1, new float[] { 1, 1 }, true, new CancellationToken(true)));
        Assert.Throws<ArgumentException>(() => Pool(new float[] { 1, 3 }, 1, new float[] { 1, 0 }, true, CancellationToken.None));
        Assert.Throws<ArgumentException>(() => Pool(new float[] { 1, 3 }, 1, new float[] { 0, 0 }, true, CancellationToken.None));
    }

    // Tiny ordinary ONNX graphs exercise the public API without downloading a model.
    sealed class Models : IDisposable
    {
        readonly string directory = Path.Combine(Path.GetTempPath(), "wespeaker-" + Guid.NewGuid().ToString("N"));
        internal string Encoder => Path.Combine(directory, "encoder.onnx");
        internal string Projection => Path.Combine(directory, "projection.onnx");
        internal Models(int frames, int width, float bad, bool wrongName)
        {
            Directory.CreateDirectory(directory);
            Save(Encoder, "fbank_features", new[] { 1, -1, 80 }, "/resnet/pool/Reshape_output_0", new[] { 1, 2560, frames },
                Enumerable.Range(0, 2560 * frames).Select(i => i == 0 ? bad : (float)(i % 7)).ToArray());
            // Slice the first 256 means. Identity with constant output would hide pooling/feeding mistakes.
            var m = New(); m.Graph.Input.Add(Value("pooled", new[] { 1, 5120 }));
            string name = wrongName ? "wrong" : "embedding"; m.Graph.Output.Add(Value(name, new[] { 1, width }));
            foreach (var (key, value) in new[] { ("starts", 0L), ("ends", (long)width), ("axes", 1L) })
            {
                var init = new TensorProto { Name = key, DataType = 7 }; init.Dims.Add(1); init.Int64Data.Add(value); m.Graph.Initializer.Add(init);
            }
            var slice = new NodeProto { OpType = "Slice" }; slice.Input.Add(new[] { "pooled", "starts", "ends", "axes" }); slice.Output.Add(name); m.Graph.Node.Add(slice);
            File.WriteAllBytes(Projection, m.ToByteArray());
        }
        static ModelProto New() => new ModelProto { IrVersion = 9, Graph = new GraphProto { Name = "fixture" },
            OpsetImport = { new OperatorSetIdProto { Domain = "", Version = 17 } } };
        static ValueInfoProto Value(string name, int[] dimensions)
        {
            var shape = new TensorShapeProto();
            foreach (int size in dimensions) shape.Dim.Add(size < 0 ? new TensorShapeProto.Types.Dimension { DimParam = "frames" }
                : new TensorShapeProto.Types.Dimension { DimValue = size });
            return new ValueInfoProto { Name = name, Type = new TypeProto { TensorType = new TypeProto.Types.Tensor { ElemType = 1, Shape = shape } } };
        }
        static void Save(string path, string input, int[] inputShape, string output, int[] outputShape, float[] data)
        {
            var m = New(); m.Graph.Input.Add(Value(input, inputShape)); m.Graph.Output.Add(Value(output, outputShape));
            var constant = new TensorProto { Name = "constant", DataType = 1 }; constant.Dims.Add(outputShape.Select(x => (long)x)); constant.FloatData.Add(data);
            m.Graph.Initializer.Add(constant);
            var node = new NodeProto { OpType = "Identity" }; node.Input.Add("constant"); node.Output.Add(output); m.Graph.Node.Add(node);
            File.WriteAllBytes(path, m.ToByteArray());
        }
        public void Dispose() => Directory.Delete(directory, true);
    }

    [Fact]
    public void PublicApiUsesNearestWeightsAndReturnsOwnedReadOnlyValues()
    {
        using var models = new Models(3, 256, 0, false); var api = new WeSpeakerEmbedder(models.Encoder, models.Projection);
        var pcm = new float[2960]; var mask = new float[] { 1, 0, 1, 0, 0 }; // 17features -> 3encoderframes; mapped indices0,1,3 => only one.
        var missing = api.Extract(pcm, 16000, mask, CancellationToken.None);
        Assert.Equal(WeSpeakerEmbeddingStatus.InsufficientFrames, missing.Status); Assert.Empty(missing.Values); Assert.Equal(1, missing.PositiveFrames);
        mask[3] = 1; // maps to weights1,0,1: channel0 values0,1,2 -> mean1.
        var result = api.Extract(pcm, 16000, mask, CancellationToken.None);
        Assert.Equal(WeSpeakerEmbeddingStatus.Completed, result.Status); Assert.Equal(2, result.PositiveFrames); Assert.Equal(3, result.EncoderFrames);
        Assert.Equal(256, result.Values.Count); Assert.Equal(1, result.Values[0]); Assert.Equal(4, result.Values[1]);
        Assert.Throws<NotSupportedException>(() => ((IList<float>)result.Values)[0] = 20);
        var before = result.Values.ToArray();
        api.Extract(pcm, 16000, Array.Empty<float>(), CancellationToken.None);
        Assert.Equal(before, result.Values); Assert.All(pcm, v => Assert.Equal(0, v)); Assert.Equal(new float[] { 1, 0, 1, 1, 0 }, mask);
    }

    [Theory]
    [InlineData(400, 1)] [InlineData(1520, 1)] [InlineData(1680, 2)]
    public void FrameBoundaryAndZeroMaskReturnExplicitStatus(int length, int frames)
    {
        using var models = new Models(frames, 256, 0, false); var api = new WeSpeakerEmbedder(models.Encoder, models.Projection);
        var unmasked = api.Extract(new float[length], 16000, Array.Empty<float>(), CancellationToken.None);
        Assert.Equal(frames, unmasked.EncoderFrames);
        Assert.Equal(frames < 2 ? WeSpeakerEmbeddingStatus.InsufficientFrames : WeSpeakerEmbeddingStatus.Completed, unmasked.Status);
        var zero = api.Extract(new float[length], 16000, new float[] { 0 }, CancellationToken.None);
        Assert.Equal(WeSpeakerEmbeddingStatus.InsufficientFrames, zero.Status); Assert.Empty(zero.Values); Assert.Equal(0, zero.PositiveFrames);
    }

    [Theory]
    [InlineData(float.NaN)] [InlineData(float.PositiveInfinity)] [InlineData(-.01f)] [InlineData(1.01f)]
    public void InvalidUnusedMaskElementIsRejectedBeforeNoData(float bad)
    {
        using var models = new Models(1, 256, 0, false); var api = new WeSpeakerEmbedder(models.Encoder, models.Projection);
        Assert.Throws<ArgumentException>(() => api.Extract(new float[400], 16000, new float[] { 0, bad }, CancellationToken.None));
    }

    [Fact]
    public void RequestErrorsCancellationAndParallelCallsPreserveRecovery()
    {
        using var models = new Models(2, 256, 0, false); var api = new WeSpeakerEmbedder(models.Encoder, models.Projection);
        var input = new float[1680]; var expected = api.Extract(input, 16000, Array.Empty<float>(), CancellationToken.None);
        Assert.Throws<OperationCanceledException>(() => api.Extract(input, 16000, Array.Empty<float>(), new CancellationToken(true)));
        Assert.Throws<ArgumentOutOfRangeException>(() => api.Extract(input, 8000, Array.Empty<float>(), CancellationToken.None));
        Assert.Throws<ArgumentOutOfRangeException>(() => api.Extract(new float[399], 16000, Array.Empty<float>(), CancellationToken.None));
        Assert.Throws<ArgumentOutOfRangeException>(() => api.Extract(new float[480001], 16000, Array.Empty<float>(), CancellationToken.None));
        Assert.Throws<ArgumentOutOfRangeException>(() => api.Extract(input, 16000, new float[480001], CancellationToken.None));
        input[^1] = float.NaN; Assert.Throws<ArgumentException>(() => api.Extract(input, 16000, new float[] { 0 }, CancellationToken.None)); input[^1] = 0;
        Parallel.For(0, 4, _ => Assert.Equal(expected.Values, api.Extract(input, 16000, Array.Empty<float>(), CancellationToken.None).Values));
        Assert.Equal(expected.Values, api.Extract(input, 16000, Array.Empty<float>(), CancellationToken.None).Values);
    }

    [Theory]
    [InlineData(1, 256, 0)] [InlineData(2, 255, 0)] [InlineData(2, 256, float.NaN)]
    public void WrongOrNonfiniteModelOutputsFail(int frames, int width, float bad)
    {
        using var models = new Models(frames, width, bad, false); var api = new WeSpeakerEmbedder(models.Encoder, models.Projection);
        Assert.Throws<InvalidDataException>(() => api.Extract(new float[1680], 16000, Array.Empty<float>(), CancellationToken.None));
    }

    [Fact]
    public void WrongGraphNamesAndMissingFilesFailAtLoad()
    {
        using var models = new Models(2, 256, 0, true);
        Assert.Throws<NotSupportedException>(() => new WeSpeakerEmbedder(models.Encoder, models.Projection));
        Assert.Throws<InvalidDataException>(() => new WeSpeakerEmbedder(models.Encoder + ".missing", models.Projection));
    }
}
