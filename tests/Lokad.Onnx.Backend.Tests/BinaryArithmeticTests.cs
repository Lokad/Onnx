using System.Numerics;

namespace Lokad.Onnx.Backend.Tests;

public class BinaryArithmeticTests
{
    [Theory]
    [InlineData(OpType.Add)]
    [InlineData(OpType.Sub)]
    [InlineData(OpType.Mul)]
    [InlineData(OpType.Div)]
    public void BroadcastDispatchPreservesEverySupportedType(OpType op)
    {
        int[] expected = op switch
        {
            OpType.Add => [10, 12, 14, 16],
            OpType.Sub => [6, 4, 10, 8],
            OpType.Mul => [16, 32, 24, 48],
            OpType.Div => [4, 2, 6, 3],
            _ => throw new InvalidOperationException()
        };
        VerifyType<sbyte>(op, expected);
        VerifyType<byte>(op, expected);
        VerifyType<short>(op, expected);
        VerifyType<ushort>(op, expected);
        VerifyType<int>(op, expected);
        VerifyType<uint>(op, expected);
        VerifyType<long>(op, expected);
        VerifyType<ulong>(op, expected);
        VerifyType<float>(op, expected);
        VerifyType<double>(op, expected);
    }

    static void VerifyType<T>(OpType op, int[] expected) where T : unmanaged, INumber<T>
    {
        var a = new DenseTensor<T>(new[] { T.CreateChecked(8), T.CreateChecked(12) }, new[] { 2, 1 });
        var b = new DenseTensor<T>(new[] { T.CreateChecked(2), T.CreateChecked(4) }, new[] { 2 });
        foreach (var options in new[] { ExecutionOptions.Scalar, ExecutionOptions.Simd })
        {
            var pool = new TensorBufferPool();
            var result = Apply(op, a, b, options, pool);
            Assert.Equal(OpStatus.Success, result.Status);
            Assert.Equal(op, result.Op);
            var output = Assert.IsType<DenseTensor<T>>(result.Outputs[0]);
            Assert.Equal(new[] { 2, 2 }, output.Dimensions.ToArray());
            Assert.Equal(expected.Select(value => T.CreateChecked(value)).ToArray(), output.ToArray());
            Assert.Equal(typeof(T) == typeof(float) && op != OpType.Sub ? 1 : 0, pool.AllocatedNew);
            Assert.Equal(new[] { T.CreateChecked(8), T.CreateChecked(12) }, a.ToArray());
            Assert.Equal(new[] { T.CreateChecked(2), T.CreateChecked(4) }, b.ToArray());
        }
    }

    [Theory]
    [InlineData(OpType.Add)]
    [InlineData(OpType.Sub)]
    [InlineData(OpType.Mul)]
    [InlineData(OpType.Div)]
    public void ValidationPrecedesOptionsAndAnyPooledRent(OpType op)
    {
        var pool = new TensorBufferPool();
        var invalid = new ExecutionOptions(OptimizationMode.Speed, new TensorExecutionOptions(false, false, 0));
        var a = DenseTensor<float>.Ones(2);
        var incompatible = DenseTensor<float>.Ones(3);
        var missing = Apply(op, null, a, invalid, pool);
        Assert.Equal(op, missing.Op);
        Assert.Equal(OpStatus.Failure, missing.Status);
        Assert.Equal("The required input parameter A is missing or null.", missing.Message);
        var wrongType = Apply(op, a, DenseTensor<int>.Ones(2), invalid, pool);
        Assert.Equal(op, wrongType.Op);
        Assert.Equal(OpStatus.Failure, wrongType.Status);
        Assert.Contains("Input tensors must be of the same type.", wrongType.Message);
        var wrongShape = Apply(op, a, incompatible, invalid, pool);
        Assert.Equal(op, wrongShape.Op);
        Assert.Equal(OpStatus.Failure, wrongShape.Status);
        Assert.Contains("not compatible for broadcasting", wrongShape.Message);
        Assert.Throws<ArgumentOutOfRangeException>(() => Apply(op, a, a, invalid, pool));
        Assert.Equal(0, pool.AllocatedNew);
        Assert.Equal(0, pool.Reused);
        Assert.Equal(0, pool.PeakOutstandingBytes);
    }

    static OpResult Apply(OpType op, ITensor? a, ITensor? b, ExecutionOptions options, TensorBufferPool pool) => op switch
    {
        OpType.Add => CPUExecutionProvider.Add(a, b, options, pool),
        OpType.Sub => CPUExecutionProvider.Sub(a, b, options),
        OpType.Mul => CPUExecutionProvider.Mul(a, b, options, pool),
        OpType.Div => CPUExecutionProvider.Div(a, b, options, pool),
        _ => throw new InvalidOperationException()
    };
}
