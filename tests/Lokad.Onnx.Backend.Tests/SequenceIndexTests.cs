namespace Lokad.Onnx.Backend.Tests;

public class SequenceIndexTests
{
    static TensorSequence Sequence() => new(new ITensor[] {
        DenseTensor<float>.Scalar(10f), DenseTensor<float>.Scalar(20f) });

    static Tensor<T>[] EmptyIndices<T>() where T : unmanaged => new Tensor<T>[] {
        new DenseTensor<T>(new[] { 0 }),
        new DenseTensor<T>(new[] { 0, 2 }, true),
        new DenseTensor<T>(new[] { 1 }).Slice(new SliceIndex(0, 0)),
        new DenseTensor<T>(new[] { 1, 0 }).BroadcastDim(0, 2) };

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EmptyIndexReturnsDescriptiveFailureAcrossLayouts(bool int64)
    {
        IEnumerable<ITensor> indices = int64 ? EmptyIndices<long>() : EmptyIndices<int>();
        foreach (var index in indices)
        {
            var result = CPUExecutionProvider.SequenceAt(Sequence(), index, null);
            Assert.Equal(OpStatus.Failure, result.Status);
            Assert.Contains("index", result.Message);
            Assert.Contains("at least one value", result.Message);
            Assert.Null(result.Cause);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GraphRejectsEmptyIndexAndRecoversOnRetry(bool int64)
    {
        var model = new OnnxModel { Name = "sequence-index" };
        model.Opset[""] = 18;
        model.Inputs.Add(new OnnxValueInfo { Name = "s", ElementType = TensorElementType.Sequence,
            SequenceElementType = TensorElementType.Float });
        model.Inputs.Add(new OnnxValueInfo { Name = "i", ElementType = int64 ? TensorElementType.Int64 : TensorElementType.Int32,
            Dims = new[] { -1 } });
        model.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float });
        model.Nodes.Add(new OnnxNode { Name = "at", OpType = "SequenceAt", Inputs = new[] { "s", "i" }, Outputs = new[] { "y" } });
        var graph = Model.Load(model);
        var inputs = new Dictionary<string, ITensor> {
            ["s"] = Sequence(),
            ["i"] = int64 ? DenseTensor<long>.OfValues(Array.Empty<long>()) : DenseTensor<int>.OfValues(Array.Empty<int>()) };
        Assert.False(graph.Execute(inputs, true));
        Assert.Equal("at", graph.LastFailedNodeName);
        Assert.Contains("at least one value", graph.LastErrorMessage);
        Assert.Null(graph.LastErrorCause);
        inputs["i"] = int64 ? DenseTensor<long>.OfValues(new[] { 1L }) : DenseTensor<int>.OfValues(new[] { 1 });
        Assert.True(graph.Execute(inputs, true), graph.LastErrorMessage);
        Assert.Equal(new[] { 20f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
        Assert.Null(graph.LastErrorCause);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NonemptyIndicesKeepFirstValueCompatibility(bool int64)
    {
        if (int64) CheckNonempty(1L, 0L, -1L);
        else CheckNonempty(1, 0, -1);
    }

    static void CheckNonempty<T>(T one, T zero, T negative) where T : unmanaged
    {
        var vector = DenseTensor<T>.OfValues(new[] { one, zero });
        var reversed = new DenseTensor<T>(new[] { 2, 2 }, true);
        reversed[0, 0] = one;
        var sliced = DenseTensor<T>.OfValues(new[] { zero, one, zero }).Slice(new SliceIndex(1, 3));
        var scalarSlice = DenseTensor<T>.OfValues(new[] { zero, one }).Slice(SliceIndex.Index(1));
        var broadcast = new DenseTensor<T>(new[] { one, zero }, new[] { 1, 2 }).BroadcastDim(0, 3);
        // Native ORT 1.29 also reads the first element of non-scalar indices.
        foreach (var index in new Tensor<T>[] { DenseTensor<T>.Scalar(one), DenseTensor<T>.Scalar(negative),
            vector, reversed, sliced, scalarSlice, broadcast })
        {
            var result = CPUExecutionProvider.SequenceAt(Sequence(), index, null);
            Assert.Equal(OpStatus.Success, result.Status);
            Assert.Equal(new[] { 20f }, ((Tensor<float>)result.Outputs[0]).ToArray());
        }
        var first = CPUExecutionProvider.SequenceAt(Sequence(), DenseTensor<T>.Scalar(zero), null);
        Assert.Equal(OpStatus.Success, first.Status);
        Assert.Equal(new[] { 10f }, ((Tensor<float>)first.Outputs[0]).ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ScalarSplitStillReadsSlicedChunkSize(bool int64)
    {
        ITensor split = int64
            ? DenseTensor<long>.OfValues(new[] { 0L, 2L }).Slice(SliceIndex.Index(1))
            : DenseTensor<int>.OfValues(new[] { 0, 2 }).Slice(SliceIndex.Index(1));
        var result = CPUExecutionProvider.SplitToSequence(DenseTensor<float>.OfValues(new[] { 10f, 20f, 30f }), split, 0, 1, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var sequence = (TensorSequence)result.Outputs[0];
        Assert.Equal(2, sequence.Items.Count);
        Assert.Equal(new[] { 10f, 20f }, ((Tensor<float>)sequence.Items[0]).ToArray());
        Assert.Equal(new[] { 30f }, ((Tensor<float>)sequence.Items[1]).ToArray());
    }
}
