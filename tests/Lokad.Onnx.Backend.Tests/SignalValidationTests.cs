using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

public class SignalValidationTests
{
    static DenseTensor<float> Signal() => new(Enumerable.Range(0, 16).Select(i => (float)i).ToArray(), new[] { 2, 8 });

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)] [InlineData(4)]
    [InlineData(5)] [InlineData(6)] [InlineData(7)] [InlineData(8)] [InlineData(9)]
    [InlineData(10)] [InlineData(11)] [InlineData(12)] [InlineData(13)] [InlineData(14)]
    [InlineData(15)] [InlineData(16)] [InlineData(17)]
    public void StftRejectsInvalidInputsBeforeWriting(int invalid)
    {
        var original = Signal();
        var before = original.ToArray();
        ITensor? signal = original, step = DenseTensor<long>.Scalar(2), window = null, length = DenseTensor<long>.Scalar(4);
        int? one = 1;
        switch (invalid)
        {
            case 0: signal = null; break;
            case 1: step = null; break;
            case 2: signal = new DenseTensor<int>(new[] { 1, 8 }); break;
            case 3: signal = new DenseTensor<float>(new[] { 8 }); break;
            case 4: signal = new DenseTensor<float>(new[] { 1, 8, 3 }); break;
            case 5: signal = new DenseTensor<float>(new[] { 1, 8, 2 }); break;
            case 6: one = 2; break;
            case 7: step = DenseTensor<float>.Scalar(1); break;
            case 8: step = DenseTensor<long>.OfValues(new long[] { 1, 2 }); break;
            case 9: step = DenseTensor<long>.Scalar(0); break;
            case 10: step = DenseTensor<long>.Scalar(-1); break;
            case 11: window = new DenseTensor<double>(new[] { 4 }); break;
            case 12: window = new DenseTensor<float>(new[] { 2, 2 }); break;
            case 13: window = new DenseTensor<float>(new[] { 3 }); break;
            case 14: length = DenseTensor<float>.Scalar(4); break;
            case 15: length = DenseTensor<long>.OfValues(new long[] { 4, 4 }); break;
            case 16: length = DenseTensor<long>.Scalar(0); break;
            case 17: length = DenseTensor<long>.Scalar(long.MaxValue); break;
        }
        Assert.Equal(OpStatus.Failure, CPU.STFT(signal, step, window, length, one, null).Status);
        Assert.Equal(before, original.ToArray());
    }

    [Fact]
    public void StftHandlesEmptyBatchAndWideStepWithoutNarrowing()
    {
        var result = CPU.STFT(new DenseTensor<float>(new[] { 0, 8 }), DenseTensor<int>.Scalar(2), null, DenseTensor<int>.Scalar(4), null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new[] { 0, 3, 3, 2 }, result.Outputs[0].Dims);
        result = CPU.STFT(Signal(), DenseTensor<long>.Scalar(long.MaxValue), null, DenseTensor<long>.Scalar(4), 0, null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new[] { 2, 1, 4, 2 }, result.Outputs[0].Dims);
        Assert.Equal(new float[] { 6, 0, -2, 2, -2, 0, -2, -2, 38, 0, -2, 2, -2, 0, -2, -2 }, ((Tensor<float>)result.Outputs[0]).ToArray());
    }

    [Fact]
    public void LogPreservesIeeeDomainBehaviorAndRejectsIntegers()
    {
        foreach (ITensor value in new ITensor[] {
            DenseTensor<float>.OfValues(new[] { 0f, -0f, -1f, float.PositiveInfinity, float.NaN, 1f }),
            DenseTensor<double>.OfValues(new[] { 0d, -0d, -1d, double.PositiveInfinity, double.NaN, 1d }) })
        {
            var result = CPU.Log(value, null);
            Assert.Equal(OpStatus.Success, result.Status);
            var output = result.Outputs[0];
            double[] actual = output is Tensor<float> f ? f.ToArray().Select(v => (double)v).ToArray() : ((Tensor<double>)output).ToArray();
            Assert.Equal(double.NegativeInfinity, actual[0]); Assert.Equal(double.NegativeInfinity, actual[1]);
            Assert.True(double.IsNaN(actual[2])); Assert.Equal(double.PositiveInfinity, actual[3]);
            Assert.True(double.IsNaN(actual[4])); Assert.Equal(0, actual[5]);
        }
        Assert.Equal(OpStatus.Failure, CPU.Log(null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Log(DenseTensor<int>.Scalar(1), null).Status);
    }

    [Fact]
    public void SumSquareRejectsBadAxesAndMissingOrUnsupportedInput()
    {
        var signal = Signal();
        Assert.Equal(OpStatus.Failure, CPU.ReduceSumSquare(null, null, null, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.ReduceSumSquare(DenseTensor<bool>.Scalar(true), null, null, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.ReduceSumSquare(signal, DenseTensor<int>.Scalar(1), null, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.ReduceSumSquare(signal, DenseTensor<float>.OfValues(new[] { 1f }), null, null, null).Status);
        Assert.Throws<ArgumentException>(() => CPU.ReduceSumSquare(signal, DenseTensor<long>.OfValues(new[] { 1L << 40 }), null, null, null));
    }

    [Fact]
    public void ReflectPadRejectsEmptyExpansionOvercropAndOverflow()
    {
        Assert.Equal(OpStatus.Failure, CPU.Pad(new DenseTensor<float>(new[] { 0 }), DenseTensor<long>.OfValues(new[] { 1L, 1L }), null, "reflect", null, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Pad(new DenseTensor<float>(new[] { 3 }), DenseTensor<long>.OfValues(new[] { -4L, 4L }), null, "reflect", null, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Pad(new DenseTensor<float>(new[] { 3 }), DenseTensor<long>.OfValues(new[] { 1L << 32, 1L }), null, "reflect", null, null, null).Status);
        var empty = CPU.Pad(new DenseTensor<float>(new[] { 3 }), DenseTensor<long>.OfValues(new[] { -3L, 0L }), null, "reflect", null, null, null);
        Assert.Equal(OpStatus.Success, empty.Status);
        Assert.Equal(0, empty.Outputs[0].Length);
    }

    [Theory]
    [InlineData(16, false)] [InlineData(17, true)] [InlineData(18, true)]
    public void StftSchemaStartsAtSeventeen(int version, bool supported)
        => Assert.Equal(supported, OperatorSchemas.IsSupported(OpType.STFT, "", version, false));
}
