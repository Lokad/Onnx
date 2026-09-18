namespace Lokad.Onnx.Backend.Tests;

public class ScaledMatMulInplaceTests
{
    static DenseTensor<float> Input(int[] dims, int seed)
    {
        var random = new Random(seed);
        int count = dims.Aggregate(1, (a, b) => a * b);
        return new DenseTensor<float>(Enumerable.Range(0, count).Select(_ => random.NextSingle() - .5f).ToArray(), dims);
    }

    static Tensor<float> Output(OpResult result)
    {
        Assert.Equal(OpStatus.Success, result.Status);
        return Assert.IsAssignableFrom<Tensor<float>>(Assert.Single(result.Outputs));
    }

    static int[] Bits(Tensor<float> tensor) => tensor.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray();

    static Tensor<float> Legacy(Tensor<float> a, Tensor<float> b, Tensor<float> divisor, ExecutionOptions options) =>
        Output(CPUExecutionProvider.Div(Output(CPUExecutionProvider.MatMul(a, b, options, null)), divisor, options, null));

    [Theory]
    [InlineData(0, false)]
    [InlineData(0, true)]
    [InlineData(1, false)]
    [InlineData(1, true)]
    [InlineData(2, false)]
    [InlineData(2, true)]
    public void LargePrivateProductPreservesDivisionBitsAndNeverReturnsEscapedStorage(int mode, bool releasePrivate)
    {
        var options = mode == 0 ? ExecutionOptions.Scalar : mode == 1 ? ExecutionOptions.Simd : ExecutionOptions.Intrinsics;
        var a = Input(new[] { 2, 128, 32 }, 61);
        var b = Input(new[] { 1, 32, 256 }, 62);
        if (mode == 2 && !System.Runtime.Intrinsics.X86.Fma.IsSupported)
        {
            Assert.Throws<InvalidOperationException>(() => CPUExecutionProvider.ScaledMatMulTrailing(
                a, b, DenseTensor<float>.Scalar(5.656854f), options, null));
            return;
        }
        var originalA = Bits(a);
        var originalB = Bits(b);
        var pool = new TensorBufferPool { ReleaseFusedTemporaries = releasePrivate };
        var poisoned = pool.Rent<float>(65536);
        Array.Fill(poisoned, float.NaN);
        pool.Return(poisoned);
        var held = new List<(Tensor<float> Tensor, int[] Bits)>();
        foreach (float d in new[] { 5.656854f, -2.5f, 0f, -0f, float.Epsilon, float.PositiveInfinity, float.NegativeInfinity, BitConverter.Int32BitsToSingle(0x7fc12345) })
        {
            var divisor = DenseTensor<float>.Scalar(d);
            var expected = Legacy(a, b, divisor, options);
            var actual = Output(CPUExecutionProvider.ScaledMatMulTrailing(a, b, divisor, options, pool));
            Assert.Equal(Bits(expected), Bits(actual));
            held.Add((actual, Bits(actual)));
            // A returned result must never be present in the free list, even
            // when release of private temporaries is independently enabled.
            var unrelated = pool.Rent<float>(65536);
            Assert.NotSame(actual.OwnedBufferArray(), unrelated);
            Array.Fill(unrelated, -12345f);
            pool.Return(unrelated);
            foreach (var item in held) Assert.Equal(item.Bits, Bits(item.Tensor));
        }
        Assert.Equal(originalA, Bits(a));
        Assert.Equal(originalB, Bits(b));
    }

    [Theory]
    [InlineData(65535)]
    [InlineData(65536)]
    public void LargeProductEliminatesTheSecondPoolRentOnlyWhenEnabled(int columns)
    {
        var a = Input(new[] { 1, 1, 3 }, 71);
        var b = Input(new[] { 1, 3, columns }, 72);
        var divisor = DenseTensor<float>.Scalar(5.656854f);
        var pool = new TensorBufferPool { ReleaseFusedTemporaries = true };
        var actual = Output(CPUExecutionProvider.ScaledMatMulTrailing(a, b, divisor, ExecutionOptions.Simd, pool));
        Assert.Equal(Bits(Legacy(a, b, divisor, ExecutionOptions.Simd)), Bits(actual));
        bool inplace = AblationSwitches.EnableScaledMatMulInplace && columns == 65536;
        Assert.Equal(inplace ? 1 : 2, pool.AllocatedNew);
        Assert.Equal(inplace ? 0 : 1, pool.Returned);
        Assert.Equal((inplace ? 1L : 2L) * columns * sizeof(float), pool.PeakOutstandingBytes);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void DivisionBroadcastShapeAndOffsetScalarRemainUnchanged(int kind)
    {
        var a = Input(new[] { 1, 256, 4 }, 81);
        var b = Input(new[] { 1, 4, 256 }, 82);
        Tensor<float> divisor = kind == 0
            ? new DenseTensor<float>(new[] { 123f, 5.656854f, -456f }.AsMemory(1, 1), new[] { 1, 1, 1 })
            : kind == 1 ? new DenseTensor<float>(new[] { 5.656854f }, new[] { 1, 1, 1, 1 })
            : Input(new[] { 256 }, 83);
        var before = Bits(divisor);
        var pool = new TensorBufferPool { ReleaseFusedTemporaries = true };
        var expected = Legacy(a, b, divisor, ExecutionOptions.Default);
        var actual = Output(CPUExecutionProvider.ScaledMatMulTrailing(a, b, divisor, ExecutionOptions.Default, pool));
        Assert.Equal(expected.Dimensions.ToArray(), actual.Dimensions.ToArray());
        Assert.Equal(Bits(expected), Bits(actual));
        Assert.Equal(before, Bits(divisor));
        Assert.Equal(AblationSwitches.EnableScaledMatMulInplace && kind == 0 ? 0 : 1, pool.Returned);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LargeDivisionFailuresStillReleaseThePrivateProduct(bool wrongType)
    {
        var a = Input(new[] { 1, 256, 4 }, 85);
        var b = Input(new[] { 1, 4, 256 }, 86);
        ITensor divisor = wrongType ? DenseTensor<int>.Scalar(2) : Input(new[] { 3 }, 87);
        var product = Output(CPUExecutionProvider.MatMul(a, b, ExecutionOptions.Simd, null));
        var expected = CPUExecutionProvider.Div(product, divisor, ExecutionOptions.Simd, null);
        var pool = new TensorBufferPool { ReleaseFusedTemporaries = true };
        var actual = CPUExecutionProvider.ScaledMatMulTrailing(a, b, divisor, ExecutionOptions.Simd, pool);
        Assert.Equal(OpStatus.Failure, actual.Status);
        Assert.Equal(expected.Message, actual.Message);
        Assert.Equal(1, pool.Returned);
        Assert.Equal(1, pool.AllocatedNew);
        Assert.Equal(65536, pool.Rent<float>(65536).Length);
        Assert.Equal(1, pool.Reused);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GraphOutputsSurviveMutationResetAndFailure(bool explicitContext)
    {
        var model = new OnnxModel { Name = "large-private-scaled-product" };
        model.Opset[""] = 14;
        model.Inputs.Add(new OnnxValueInfo { Name = "a", ElementType = TensorElementType.Float, Dims = new[] { 1, 256, 4 } });
        model.Inputs.Add(new OnnxValueInfo { Name = "b", ElementType = TensorElementType.Float, Dims = new[] { 1, 4, 256 } });
        model.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { 1, 256, 256 } });
        model.Initializers.Add(new OnnxTensor { Name = "d", ElementType = TensorElementType.Float, Dims = Array.Empty<int>(), Data = new[] { 5.656854f } });
        model.Nodes.Add(new OnnxNode { Name = "product", OpType = "MatMul", Inputs = new[] { "a", "b" }, Outputs = new[] { "p" } });
        model.Nodes.Add(new OnnxNode { Name = "scale", OpType = "Div", Inputs = new[] { "p", "d" }, Outputs = new[] { "y" } });
        var graph = Model.Load(model) ?? throw new InvalidOperationException("Model did not load");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.ScaledMatMul);
        ComputationalGraph execution = explicitContext ? graph.CreateExecution(null) : graph;
        var a = Input(new[] { 1, 256, 4 }, 91);
        var b = Input(new[] { 1, 4, 256 }, 92);
        var held = new List<(Tensor<float> Tensor, int[] Bits)>();
        for (int call = 0; call < 3; call++)
        {
            b.Buffer.Span[7] += .25f;
            var beforeA = Bits(a);
            var beforeB = Bits(b);
            var expected = Legacy(a, b, DenseTensor<float>.Scalar(5.656854f), ExecutionOptions.Default);
            Assert.True(execution.Execute(new Dictionary<string, ITensor> { ["a"] = a, ["b"] = b }, false), execution.LastErrorMessage);
            var actual = Assert.IsAssignableFrom<Tensor<float>>(execution.Outputs["y"]);
            Assert.Equal(Bits(expected), Bits(actual));
            held.Add((actual, Bits(actual)));
            execution.Reset();
            Assert.False(execution.Execute(new Dictionary<string, ITensor>(), false));
            foreach (var item in held) Assert.Equal(item.Bits, Bits(item.Tensor));
            Assert.Equal(beforeA, Bits(a));
            Assert.Equal(beforeB, Bits(b));
        }
    }
}
