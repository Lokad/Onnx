namespace Lokad.Onnx.Backend.Tests;

public class EmptySequenceTests
{
    public static IEnumerable<object[]> DefaultAndScalarSplits()
    {
        foreach (var type in new[] { TensorElementType.Float, TensorElementType.Double, TensorElementType.Int32,
            TensorElementType.Int64, TensorElementType.UInt32, TensorElementType.UInt64, TensorElementType.Bool })
        foreach (int splitKind in new[] { 0, 1, 2 })
        foreach (int keepdims in new[] { 0, 1 })
            yield return new object[] { type, splitKind, keepdims };
    }

    static ITensor EmptyInput(TensorElementType type) => type switch
    {
        TensorElementType.Float => DenseTensor<float>.OfShape(0, 2),
        TensorElementType.Double => DenseTensor<double>.OfShape(0, 2),
        TensorElementType.Int32 => DenseTensor<int>.OfShape(0, 2),
        TensorElementType.Int64 => DenseTensor<long>.OfShape(0, 2),
        TensorElementType.UInt32 => DenseTensor<uint>.OfShape(0, 2),
        TensorElementType.UInt64 => DenseTensor<ulong>.OfShape(0, 2),
        TensorElementType.Bool => DenseTensor<bool>.OfShape(0, 2),
        _ => throw new ArgumentOutOfRangeException(nameof(type))
    };

    [Theory]
    [MemberData(nameof(DefaultAndScalarSplits))]
    public void EmptySplitAxisProducesNoItems(TensorElementType type, int splitKind, int keepdims)
    {
        // ORT 1.29: an empty split axis yields no pieces for absent/scalar
        // splits, independently of keepdims. Explicit vector [0] differs.
        var input = EmptyInput(type);
        ITensor? split = splitKind == 0 ? null : splitKind == 1 ? DenseTensor<int>.Scalar(1) : DenseTensor<long>.Scalar(1);
        var result = CPUExecutionProvider.SplitToSequence(input, split, 0, keepdims, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var sequence = Assert.IsType<TensorSequence>(Assert.Single(result.Outputs));
        Assert.Empty(sequence.Items);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.SequenceAt(sequence, DenseTensor<long>.Scalar(0), null).Status);
        Assert.Equal(new[] { 0, 2 }, input.Dims);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void ExplicitZeroSizeStillProducesOneEmptyItem(int keepdims)
    {
        var result = CPUExecutionProvider.SplitToSequence(DenseTensor<float>.OfShape(0, 2),
            DenseTensor<long>.OfValues(new long[] { 0 }), 0, keepdims, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var item = Assert.Single(((TensorSequence)Assert.Single(result.Outputs)).Items);
        Assert.Equal(new[] { 0, 2 }, item.Dims);
        Assert.Empty(item.ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void NonemptySplitAxisStillProducesPiecesWhenAnotherAxisIsEmpty(int keepdims)
    {
        var result = CPUExecutionProvider.SplitToSequence(DenseTensor<float>.OfShape(0, 2), null, 1, keepdims, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var items = ((TensorSequence)Assert.Single(result.Outputs)).Items;
        Assert.Equal(2, items.Count);
        Assert.All(items, item =>
        {
            Assert.Equal(keepdims == 1 ? new[] { 0, 1 } : new[] { 0 }, item.Dims);
            Assert.Empty(item.ToArray());
        });
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(0, 1)]
    [InlineData(1, 0)]
    [InlineData(1, 1)]
    [InlineData(2, 0)]
    [InlineData(2, 1)]
    public void SequenceOutputsSurviveGraphReuse(int splitKind, int keepdims)
    {
        var model = new OnnxModel { Name = "empty-sequence" };
        model.Opset[""] = 18;
        model.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = [0, 2] });
        model.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Sequence });
        if (splitKind != 0)
            model.Initializers.Add(new OnnxTensor { Name = "s", ElementType = TensorElementType.Int64,
                Dims = splitKind == 1 ? [] : [1], Data = splitKind == 1 ? new long[] { 1 } : new long[] { 0 } });
        model.Nodes.Add(new OnnxNode { OpType = "SplitToSequence", Inputs = splitKind == 0 ? ["x"] : ["x", "s"],
            Outputs = ["y"], Attributes = new() { ["axis"] = 0, ["keepdims"] = keepdims } });
        var graph = Model.Load(model);
        var input = DenseTensor<float>.OfShape(0, 2);
        Assert.True(graph.Execute(new ITensor[] { input }, true), graph.LastErrorMessage);
        var held = Assert.IsType<TensorSequence>(graph.Outputs["y"]);
        Assert.Equal(splitKind == 2 ? 1 : 0, held.Items.Count);
        Assert.True(graph.Execute(new ITensor[] { input }, true), graph.LastErrorMessage);
        var current = Assert.IsType<TensorSequence>(graph.Outputs["y"]);
        Assert.NotSame(held, current);
        Assert.Equal(splitKind == 2 ? 1 : 0, current.Items.Count);
        Assert.Equal(splitKind == 2 ? 1 : 0, held.Items.Count);
        Assert.All(held.Items, item => Assert.Equal(new[] { 0, 2 }, item.Dims));
    }
}
