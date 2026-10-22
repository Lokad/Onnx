using System;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using Xunit;

namespace Lokad.Onnx.Tensors.Tests;

public class ReduceMeanOptionsTests
{
    static ITensor Reduce(string dtype, bool noOp, TensorExecutionOptions options) => dtype switch
    {
        "int" => Tensor<int>.ReduceMean(DenseTensor<int>.OfValues(new[] { 1, 2 }), null, false, noOp, options),
        "long" => Tensor<long>.ReduceMean(DenseTensor<long>.OfValues(new long[] { 1, 2 }), null, false, noOp, options),
        "float" => Tensor<float>.ReduceMean(DenseTensor<float>.OfValues(new float[] { 1, 2 }), null, false, noOp, options),
        "double" => Tensor<double>.ReduceMean(DenseTensor<double>.OfValues(new double[] { 1, 2 }), null, false, noOp, options),
        _ => throw new ArgumentOutOfRangeException(nameof(dtype))
    };

    [Theory]
    [InlineData("int", false)]
    [InlineData("int", true)]
    [InlineData("long", false)]
    [InlineData("long", true)]
    [InlineData("float", false)]
    [InlineData("float", true)]
    [InlineData("double", false)]
    [InlineData("double", true)]
    public void InvalidOptions_AreRejectedBeforeReductionOrNoop(string dtype, bool noOp)
    {
        foreach (int workers in new[] { -1, 0 })
        {
            var error = Assert.Throws<ArgumentOutOfRangeException>(() => Reduce(dtype, noOp, new(false, false, workers)));
            Assert.Equal("MaxDegreeOfParallelism", error.ParamName);
        }
        var flags = Assert.Throws<ArgumentException>(() => Reduce(dtype, noOp, new(false, true, 1)));
        Assert.Equal("UseIntrinsics", flags.ParamName);
    }

    [SkippableTheory]
    [InlineData("int")]
    [InlineData("long")]
    [InlineData("float")]
    [InlineData("double")]
    public void ExplicitUnsupportedIntrinsics_AreRejectedBeforeAllPaths(string dtype)
    {
        Skip.If(Fma.IsSupported, "Run with DOTNET_EnableHWIntrinsic=0 to exercise unsupported FMA.");
        foreach (bool noOp in new[] { false, true })
            Assert.Throws<InvalidOperationException>(() => Reduce(dtype, noOp, TensorExecutionOptions.Intrinsics));
    }

    [Theory]
    [InlineData("int", false)]
    [InlineData("int", true)]
    [InlineData("long", false)]
    [InlineData("long", true)]
    [InlineData("float", false)]
    [InlineData("float", true)]
    [InlineData("double", false)]
    [InlineData("double", true)]
    public void ValidOptions_PreserveResults(string dtype, bool noOp)
    {
        double mean = dtype is "int" or "long" ? 1 : 1.5;
        double[] expected = noOp ? [1, 2] : [mean];
        foreach (var options in new[] { TensorExecutionOptions.Scalar, TensorExecutionOptions.Simd, TensorExecutionOptions.Auto })
        {
            var result = Reduce(dtype, noOp, options);
            Assert.Equal(expected, result.ToArray().Cast<object>().Select(Convert.ToDouble).ToArray());
            Assert.Equal(noOp ? new[] { 2 } : Array.Empty<int>(), result.Dims);
        }
    }
}
