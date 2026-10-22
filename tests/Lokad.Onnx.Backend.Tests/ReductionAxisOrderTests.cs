namespace Lokad.Onnx.Backend.Tests;

[Collection("NonParallelAllocation")]
public class ReductionAxisOrderTests
{
    static int[] Axes(int variant) => variant switch
    {
        0 => [0, 1], 1 => [1, 0], 2 => [-1, -2], 3 => [1, 0, -1],
        _ => throw new ArgumentOutOfRangeException(nameof(variant))
    };

    [Theory]
    [InlineData(0, false)]
    [InlineData(1, false)]
    [InlineData(2, false)]
    [InlineData(3, false)]
    [InlineData(0, true)]
    [InlineData(1, true)]
    [InlineData(2, true)]
    [InlineData(3, true)]
    public void EquivalentAxisSets_PreserveNativeSum(int variant, bool reversed)
    {
        var data = new DenseTensor<float>(new float[4], new[] { 2, 2 }, reversed);
        data[0, 0] = 100000000; data[0, 1] = 1; data[1, 0] = -100000000; data[1, 1] = 1;
        var axes = DenseTensor<int>.OfValues(Axes(variant));
        float[] before = data.ToArray();
        foreach (bool keep in new[] { false, true })
        {
            // ORT 1.29 CPU gives 2 for every permutation/duplicate of this set.
            var sum = Tensor<float>.ReduceSum(data, axes, keep, false);
            Assert.Equal(new float[] { 2 }, sum.ToArray());
            Assert.Equal(keep ? new[] { 1, 1 } : Array.Empty<int>(), sum.Dimensions.ToArray());
            Assert.False(TensorAlias.SharesBackingMemory(data, sum));
        }
        Assert.Equal(before, data.ToArray());
        Assert.Equal(Axes(variant), axes.ToArray());
    }

    [Fact]
    public void ReversedAxisSet_DoesNotTransposeTheWholePayload()
    {
        var data = new DenseTensor<float>(Enumerable.Repeat(1f, 128 * 256).ToArray(), new[] { 128, 256 });
        var axes = DenseTensor<int>.OfValues(new[] { 1, 0 });
        for (int i = 0; i < 3; i++) Tensor<float>.ReduceSum(data, axes, false, false);
        long before = GC.GetAllocatedBytesForCurrentThread();
        var result = Tensor<float>.ReduceSum(data, axes, false, false);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(new float[] { 32768 }, result.ToArray());
        Assert.True(allocated < 4096, $"Full reduction allocated {allocated} bytes for a 131072-byte payload.");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PartialAxes_PreserveKeptDimensionsAndOtherDtypes(bool keep)
    {
        var axes = DenseTensor<int>.OfValues(new[] { -1, 0, -1 });
        int[] shape = keep ? [1, 3, 1] : [3];
        var ints = new DenseTensor<int>(Enumerable.Range(0, 12).ToArray(), new[] { 2, 3, 2 });
        var doubles = new DenseTensor<double>(Enumerable.Range(0, 12).Select(i => (double)i).ToArray(), new[] { 2, 3, 2 });
        var sum = Tensor<int>.ReduceSum(ints, axes, keep, false);
        var mean = Tensor<double>.ReduceMean(doubles, axes, keep, false);
        var max = Tensor<int>.ReduceMax(ints, axes, keep, false);
        Assert.Equal(new[] { 14, 22, 30 }, sum.ToArray());
        Assert.Equal(new double[] { 3.5, 5.5, 7.5 }, mean.ToArray());
        Assert.Equal(new[] { 7, 9, 11 }, max.ToArray());
        Assert.Equal(shape, sum.Dimensions.ToArray());
        Assert.Equal(shape, mean.Dimensions.ToArray());
        Assert.Equal(shape, max.Dimensions.ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void ImportedGraph_PreservesAxisSetNumerics(int variant)
    {
        var model = new OnnxModel { Name = "axis-order" };
        model.Opset[""] = 18;
        model.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = [2, 2] });
        model.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = [] });
        model.Initializers.Add(new OnnxTensor { Name = "a", ElementType = TensorElementType.Int64,
            Dims = [Axes(variant).Length], Data = Axes(variant).Select(v => (long)v).ToArray() });
        model.Nodes.Add(new OnnxNode { Name = "sum", OpType = "ReduceSum", Inputs = ["x", "a"], Outputs = ["y"], Attributes = new() { ["keepdims"] = 0 } });
        var graph = Model.Load(model);
        var input = DenseTensor<float>.OfValues(new float[,] { { float.MaxValue, float.MaxValue }, { -float.MaxValue, -float.MaxValue } });
        foreach (var options in new[] { ExecutionOptions.Default, ExecutionOptions.Scalar, ExecutionOptions.Simd })
        {
            Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true, ExecutionProvider.CPU, options), graph.LastErrorMessage);
            Assert.Equal(new float[] { 0 }, ((Tensor<float>)graph.Outputs["y"]!).ToArray());
        }
    }
}
