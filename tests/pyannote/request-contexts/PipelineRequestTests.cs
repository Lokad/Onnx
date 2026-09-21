namespace Lokad.Onnx.Backend.Tests;

using System.Reflection;
using System.Runtime.ExceptionServices;
using Google.Protobuf;
using global::Onnx;

public class PipelineRequestTests
{
    const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;
    static object Invoke(object target, string method, params object?[] args)
    {
        try { return target.GetType().GetMethod(method, Private)!.Invoke(target, args)!; }
        catch (TargetInvocationException e) when (e.InnerException is not null)
        { ExceptionDispatchInfo.Capture(e.InnerException).Throw(); throw; }
    }
    static IDisposable Request(WeSpeakerEmbedder api) => (IDisposable)Invoke(api, "CreatePipelineRequest");
    static float[][] Masks() => Enumerable.Range(0, 3).Select(_ => Enumerable.Repeat(1f, 589).ToArray()).ToArray();
    static float[] Pcm(float gain) => Enumerable.Range(0, 160000).Select(i => gain * (float)Math.Sin(2 * Math.PI * 437 * i / 16000)).ToArray();
    static WeSpeakerEmbedding[] Extract(object request, float[] samples, float[][] masks, CancellationToken token = default) =>
        (WeSpeakerEmbedding[])Invoke(request, "ExtractPipeline", samples, masks, token);
    static float[][] Snapshot(WeSpeakerEmbedding[] values) => values.Select(v => v.Values.ToArray()).ToArray();
    static void Equal(float[][] expected, WeSpeakerEmbedding[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i].Values);
    }

    [Fact]
    public void ChangingWindowsAndInterleavedRequestsPreserveOwnedResults()
    {
        using var models = new Models(); var api = models.Create(); var a = Pcm(.1f); var b = Pcm(.3f); var masks = Masks();
        var beforeA = a.ToArray(); var beforeB = b.ToArray();
        var standaloneBefore = api.Extract(a, 16000, Array.Empty<float>(), CancellationToken.None).Values.ToArray();
        using var first = Request(api); using var second = Request(api);
        var held = Extract(first, a, masks); var snapshot = Snapshot(held);
        var changed = Extract(first, b, masks);
        Assert.NotEqual(snapshot[0][0], changed[0].Values[0]);
        Equal(snapshot, held);
        Equal(Snapshot(changed), Extract(second, b, masks));
        Equal(snapshot, Extract(second, a, masks));
        Equal(snapshot, Extract(first, a, masks));
        first.Dispose(); second.Dispose(); Equal(snapshot, held);
        Assert.Equal(beforeA, a); Assert.Equal(beforeB, b);
        foreach (var mask in masks) Assert.All(mask, value => Assert.Equal(1f, value));
        Assert.Throws<NotSupportedException>(() => ((IList<float>)held[0].Values)[0] = 0);
        var standalone = api.Extract(a, 16000, Array.Empty<float>(), CancellationToken.None);
        Assert.Equal(standaloneBefore, standalone.Values);
    }

    [Fact]
    public void CancellationInvalidInputAndGraphFailureRecoverWithExistingContexts()
    {
        using var models = new Models(); var api = models.Create(); var pcm = Pcm(.2f); var masks = Masks();
        using var request = Request(api); var held = Extract(request, pcm, masks); var expected = Snapshot(held);
        Assert.Throws<OperationCanceledException>(() => Extract(request, pcm, masks, new CancellationToken(true)));
        var bad = Pcm(.2f); bad[17] = float.NaN;
        Assert.Throws<ArgumentException>(() => Extract(request, bad, masks));
        masks[0][12] = .5f;
        Assert.Throws<ArgumentException>(() => Extract(request, pcm, masks)); masks[0][12] = 1;
        // Fault injection changes a graph parameter only while no inference is active.
        // The Slice then returns a wrong output shape after encoder execution has succeeded.
        var graph = (ComputationalGraph)typeof(WeSpeakerEmbedder).GetField("projection", Private)!.GetValue(api)!;
        var end = graph.Initializers["ends"];
        try
        {
            graph.Initializers["ends"] = new DenseTensor<long>(new long[] { 0 }, new[] { 1 });
            Assert.Throws<InvalidDataException>(() => Extract(request, pcm, masks));
        }
        finally { graph.Initializers["ends"] = end; }
        Equal(expected, Extract(request, pcm, masks)); Equal(expected, held);
        request.Dispose();
        Assert.Null(request.GetType().GetField("Encoding", Private)!.GetValue(request));
        Assert.Null(request.GetType().GetField("Projecting", Private)!.GetValue(request));
        Assert.Throws<ObjectDisposedException>(() => Extract(request, pcm, masks));
        using var recovered = Request(api); Equal(expected, Extract(recovered, pcm, masks)); Equal(expected, held);
    }

    sealed class Models : IDisposable
    {
        readonly string directory = Path.Combine(Path.GetTempPath(), "pipeline-request-" + Guid.NewGuid().ToString("N"));
        internal Models()
        {
            Directory.CreateDirectory(directory);
            var encoder = New(); encoder.Graph.Input.Add(Value("fbank_features", 1, 998, 80));
            encoder.Graph.Output.Add(Value("/resnet/pool/Reshape_output_0", 1, 2560, 125));
            var mean = new NodeProto { OpType = "ReduceMean" }; mean.Input.Add("fbank_features"); mean.Output.Add("mean");
            var axes = new AttributeProto { Name = "axes", Type = AttributeProto.Types.AttributeType.Ints }; axes.Ints.Add(new long[] { 1, 2 }); mean.Attribute.Add(axes);
            mean.Attribute.Add(new AttributeProto { Name = "keepdims", Type = AttributeProto.Types.AttributeType.Int, I = 1 }); encoder.Graph.Node.Add(mean);
            var basis = new TensorProto { Name = "basis", DataType = 1 }; basis.Dims.Add(new long[] { 1, 2560, 125 });
            basis.FloatData.Add(Enumerable.Range(0, 2560 * 125).Select(i => (float)(i % 7))); encoder.Graph.Initializer.Add(basis);
            var add = new NodeProto { OpType = "Add" }; add.Input.Add(new[] { "mean", "basis" }); add.Output.Add("/resnet/pool/Reshape_output_0"); encoder.Graph.Node.Add(add);
            File.WriteAllBytes(Path.Combine(directory, "encoder.onnx"), encoder.ToByteArray());
            var projection = New(); projection.Graph.Input.Add(Value("pooled", 1, 5120)); projection.Graph.Output.Add(Value("embedding", 1, 256));
            foreach (var (name, value) in new[] { ("starts", 0L), ("ends", 256L), ("axes", 1L) })
            {
                var tensor = new TensorProto { Name = name, DataType = 7 }; tensor.Dims.Add(1); tensor.Int64Data.Add(value); projection.Graph.Initializer.Add(tensor);
            }
            var slice = new NodeProto { OpType = "Slice" }; slice.Input.Add(new[] { "pooled", "starts", "ends", "axes" }); slice.Output.Add("embedding"); projection.Graph.Node.Add(slice);
            File.WriteAllBytes(Path.Combine(directory, "projection.onnx"), projection.ToByteArray());
        }
        internal WeSpeakerEmbedder Create() => new(Path.Combine(directory, "encoder.onnx"), Path.Combine(directory, "projection.onnx"));
        static ModelProto New() => new() { IrVersion = 9, Graph = new GraphProto { Name = "input-dependent-fixture" },
            OpsetImport = { new OperatorSetIdProto { Domain = "", Version = 17 } } };
        static ValueInfoProto Value(string name, params int[] dimensions)
        {
            var shape = new TensorShapeProto(); foreach (int size in dimensions) shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = size });
            return new ValueInfoProto { Name = name, Type = new TypeProto { TensorType = new TypeProto.Types.Tensor { ElemType = 1, Shape = shape } } };
        }
        public void Dispose() => Directory.Delete(directory, true);
    }
}
