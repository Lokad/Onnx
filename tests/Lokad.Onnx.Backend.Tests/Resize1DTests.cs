namespace Lokad.Onnx.Backend.Tests;

public class Resize1DTests
{
    static OpResult Resize(ITensor input, ITensor? scales = null, ITensor? sizes = null, string mode = "nearest") =>
        CPUExecutionProvider.Resize(input, null, scales, sizes, mode, "asymmetric", "floor", null, null, null);

    static Tensor<float> Output(OpResult result)
    {
        Assert.True(result.Status == OpStatus.Success, result.Message);
        return Assert.IsAssignableFrom<Tensor<float>>(Assert.Single(result.Outputs));
    }

    [Fact]
    public void NearestUpsamplingPreservesBatchesAndChannels()
    {
        var values = Enumerable.Range(1, 12).Select(x => (float)x).ToArray();
        var input = new DenseTensor<float>(values, new[] { 2, 2, 3 });
        var result = Output(Resize(input, DenseTensor<float>.OfValues(new[] { 1f, 1f, 2f })));
        Assert.Equal(new[] { 2, 2, 6 }, result.Dimensions.ToArray());
        Assert.Equal(values.SelectMany(v => new[] { v, v }), result.ToArray());
    }

    [Fact]
    public void NonintegerScalesRetainOriginalSamplingRatherThanRecomputedRatio()
    {
        var input = new DenseTensor<float>(new[] { 1f, 2f, 3f }, new[] { 1, 1, 3 });
        var scaled = Output(Resize(input, DenseTensor<float>.OfValues(new[] { 1f, 1f, 1.6f })));
        Assert.Equal(new[] { 1f, 1f, 2f, 2f }, scaled.ToArray());
        var sized = Output(Resize(input, sizes: DenseTensor<long>.OfValues(new long[] { 1, 1, 4 })));
        Assert.Equal(new[] { 1f, 1f, 2f, 3f }, sized.ToArray());
        Assert.Equal(0, Output(Resize(new DenseTensor<float>(new[] { 1, 1, 0 }), DenseTensor<float>.OfValues(new[] { 1f, 1f, 2f }))).Length);
    }

    [Fact]
    public void RejectsChangedChannelsBadScaleArityAndUnimplementedLinearMode()
    {
        var input = new DenseTensor<float>(new[] { 1, 2, 3 });
        Assert.Equal(OpStatus.Failure, Resize(input, DenseTensor<float>.OfValues(new[] { 1f, 2f, 2f })).Status);
        Assert.Equal(OpStatus.Failure, Resize(input, DenseTensor<float>.OfValues(new[] { 1f, 2f })).Status);
        Assert.Equal(OpStatus.Failure, Resize(input, DenseTensor<float>.OfValues(new[] { 1f, 1f, 2f }), mode: "linear").Status);
        Assert.Equal(OpStatus.Failure, Resize(input, sizes: DenseTensor<long>.OfValues(new long[] { 1, 2, 0 })).Status);
    }
}
