namespace Lokad.Onnx.Backend.Tests;

public class ConvTranspose1DTests
{
    static OpResult Conv(ITensor input, ITensor weights, ITensor? bias = null, int group = 1,
        int stride = 1, int dilation = 1, int left = 0, int right = 0, int padding = 0) =>
        CPUExecutionProvider.ConvTranspose(input, weights, bias, null, new[] { dilation }, group,
            null, new[] { left, right }, new[] { stride }, new[] { padding }, null, null);

    static Tensor<T> Output<T>(OpResult result) where T : unmanaged
    {
        Assert.True(result.Status == OpStatus.Success, result.Message);
        return Assert.IsAssignableFrom<Tensor<T>>(Assert.Single(result.Outputs));
    }

    [Fact]
    public void OverlapPaddingAndBiasMatchSmallOrtExample()
    {
        var input = new DenseTensor<float>(new[] { 1f, 2f, 3f }, new[] { 1, 1, 3 });
        var weights = new DenseTensor<float>(new[] { 1f, 2f, 1f }, new[] { 1, 1, 3 });
        var result = Output<float>(Conv(input, weights, DenseTensor<float>.OfValues(new[] { .5f }), stride: 2, left: 1, right: 1, padding: 1));
        Assert.Equal(new[] { 1, 1, 6 }, result.Dimensions.ToArray());
        Assert.Equal(new[] { 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 3.5f }, result.ToArray());
    }

    [Theory]
    [InlineData(1, 1, 1, 0, 0, 0)]
    [InlineData(2, 2, 1, 1, 1, 1)]
    [InlineData(2, 2, 2, 1, 2, 1)]
    [InlineData(4, 1, 2, 2, 1, 1)]
    public void GroupsBatchesDilationAndViewsMatchIndependentOutputGatherOracle(int groups, int stride, int dilation, int left, int right, int padding)
    {
        const int batch = 2, channels = 4, length = 5, outputsPerGroup = 2, kernel = 3;
        var denseInput = new DenseTensor<float>(Enumerable.Range(0, batch * length * channels).Select(i => (i % 11 - 5) * .25f).ToArray(), new[] { batch, length, channels });
        var input = Output<float>(CPUExecutionProvider.Transpose(denseInput, new[] { 0, 2, 1 }, null, null));
        var denseWeights = new DenseTensor<float>(Enumerable.Range(0, channels * outputsPerGroup * kernel).Select(i => (i % 7 - 3) * .5f).ToArray(), new[] { outputsPerGroup, channels, kernel });
        var weights = Output<float>(CPUExecutionProvider.Transpose(denseWeights, new[] { 1, 0, 2 }, null, null));
        int outputChannels = groups * outputsPerGroup;
        var bias = DenseTensor<float>.OfValues(Enumerable.Range(0, outputChannels).Select(i => (i - 2) * .125f).ToArray());
        var actual = Output<float>(Conv(input, weights, bias, groups, stride, dilation, left, right, padding));
        int outputLength = (length - 1) * stride - left - right + dilation * (kernel - 1) + padding + 1;
        Assert.Equal(new[] { batch, outputChannels, outputLength }, actual.Dimensions.ToArray());
        var x = input.ToArray();
        var w = weights.ToArray();
        var expected = new List<float>();
        for (int n = 0; n < batch; n++)
            for (int oc = 0; oc < outputChannels; oc++)
                for (int position = 0; position < outputLength; position++)
                {
                    // Gather possible source samples from an output coordinate, independent
                    // of the production kernel's scatter traversal. All values are exact binary fractions.
                    float sum = bias.GetValue(oc);
                    int group = oc / outputsPerGroup, perGroup = channels / groups;
                    for (int ic = group * perGroup; ic < (group + 1) * perGroup; ic++)
                        for (int tap = 0; tap < kernel; tap++)
                        {
                            int source = position + left - tap * dilation;
                            if (source >= 0 && source % stride == 0 && source / stride < length)
                                sum += x[(n * channels + ic) * length + source / stride] * w[(ic * outputsPerGroup + oc % outputsPerGroup) * kernel + tap];
                        }
                    expected.Add(sum);
                }
        Assert.Equal(expected, actual.ToArray());
        var doubleInput = new DenseTensor<double>(x.Select(v => (double)v).ToArray(), input.Dimensions.ToArray());
        var doubleWeights = new DenseTensor<double>(w.Select(v => (double)v).ToArray(), weights.Dimensions.ToArray());
        var doubleBias = DenseTensor<double>.OfValues(bias.ToArray().Select(v => (double)v).ToArray());
        Assert.Equal(expected.Select(v => (double)v), Output<double>(Conv(doubleInput, doubleWeights, doubleBias, groups, stride, dilation, left, right, padding)).ToArray());
    }

    [Fact]
    public void ValidatesChannelsPaddingRankAndUnimplementedOutputShape()
    {
        var x = new DenseTensor<float>(new[] { 1, 4, 3 });
        var w = new DenseTensor<float>(new[] { 4, 1, 3 });
        Assert.Equal(OpStatus.Failure, Conv(x, w, group: 3).Status);
        Assert.Equal(OpStatus.Failure, Conv(x, w, padding: 1).Status);
        Assert.Equal(OpStatus.Failure, Conv(x, w, left: -1).Status);
        Assert.Equal(OpStatus.Failure, Conv(x, w, DenseTensor<float>.OfValues(new[] { 1f, 2f })).Status);
        Assert.Equal(OpStatus.Failure, Conv(new DenseTensor<float>(new[] { 1, 4, 1, 3 }), w).Status);
        var explicitShape = CPUExecutionProvider.ConvTranspose(x, w, null, null, null, 1, null, null, null, null, new[] { 5 }, null);
        Assert.Equal(OpStatus.Failure, explicitShape.Status);
        Assert.Contains("output_shape", explicitShape.Message);
        var emptyBatch = new DenseTensor<float>(new[] { 0, 1, 65536 });
        Assert.Equal(OpStatus.Failure, Conv(emptyBatch, new DenseTensor<float>(new[] { 1, 1, 3 }), stride: int.MaxValue).Status);
    }

    [Fact]
    public void MatrixReconstructionMatchesScalarWithDilationBiasAndChangedWeights()
    {
        var input = new DenseTensor<float>(Enumerable.Range(0, 64).Select(i => (i % 5 - 2) * .25f).ToArray(), new[] { 1, 16, 4 });
        var weights = new DenseTensor<float>(Enumerable.Range(0, 16 * 2 * 17).Select(i => (i % 7 - 3) * .125f).ToArray(), new[] { 16, 2, 17 });
        var bias = DenseTensor<float>.OfValues(new[] { .5f, -.25f });
        OpResult Execute(ExecutionOptions options) => CPUExecutionProvider.ConvTranspose(input, weights, bias, null,
            new[] { 2 }, 1, null, new[] { 1, 2 }, new[] { 2 }, new[] { 1 }, null, options);
        for (int revision = 0; revision < 2; revision++)
        {
            var expected = Output<float>(Execute(ExecutionOptions.Scalar));
            var actual = Output<float>(Execute(ExecutionOptions.Simd));
            Assert.Equal(new[] { 1, 2, 37 }, actual.Dimensions.ToArray());
            Assert.Equal(expected.ToArray(), actual.ToArray());
            weights.SetValue(0, weights.GetValue(0) + 1f);
        }
    }

    [Fact]
    public void SchemaDispatchExecutesAndCallerOwnedOutputsSurviveAnotherCall()
    {
        var graph = new ComputationalGraph { Opset = new() { [""] = 17 } };
        graph.Inputs["x"] = new DenseTensor<float>(new[] { 1f, 2f }, new[] { 1, 1, 2 });
        graph.Inputs["w"] = new DenseTensor<float>(new[] { 1f, 2f }, new[] { 1, 1, 2 });
        var node = new Node { Name = "transpose-conv", Op = OpType.ConvTranspose, OpTypeName = "ConvTranspose",
            Inputs = new[] { "x", "w" }, Outputs = new[] { "y" }, Attributes = new(), OpsetVersion = 17 };
        Assert.True(CPUExecutionProvider.SupportsNode(node));
        var first = Output<float>(node.Execute(graph, ExecutionProvider.CPU, null));
        Assert.Equal(new[] { 1f, 4f, 4f }, first.ToArray());
        _ = Output<float>(node.Execute(graph, ExecutionProvider.CPU, null));
        Assert.Equal(new[] { 1f, 4f, 4f }, first.ToArray());
    }
}
