using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins integer edge behavior against ORT 1.29 probe values: Add, Sub, and Mul
/// overflow wraps modulo the width, while integer division errors fail cleanly
/// with node identity. Min-by-negative-one throws through checked generic-math
/// division where the native reference itself aborts the process (0xC0000095),
/// so clean failure without a fault is the contract.
/// </summary>
public class IntegerArithmeticBoundaryTests
{
    static ITensor Apply(OpType op, ITensor a, ITensor b)
    {
        var result = op switch
        {
            OpType.Add => CPU.Add(a, b, null, null),
            OpType.Sub => CPU.Sub(a, b, null),
            OpType.Mul => CPU.Mul(a, b, null, null),
            _ => throw new ArgumentOutOfRangeException(nameof(op)),
        };
        Assert.Equal(OpStatus.Success, result.Status);
        return result.Outputs![0];
    }

    static ComputationalGraph BinaryGraph(OpType op, ITensor x, ITensor y, ITensor z)
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        g.Inputs["x"] = x;
        g.Inputs["y"] = y;
        g.Outputs["z"] = z;
        g.Nodes.Add(new Node { Name = "n", Op = op, Inputs = new[] { "x", "y" }, Outputs = new[] { "z" } });
        return g;
    }

    static Dictionary<string, ITensor> Pair(ITensor x, ITensor y) =>
        new Dictionary<string, ITensor> { { "x", x }, { "y", y } };

    [Fact]
    public void Overflow_WrapsModuloWidth()
    {
        var add = (Tensor<int>)Apply(OpType.Add, DenseTensor<int>.OfValues(new int[] { 2147483647 }), DenseTensor<int>.OfValues(new int[] { 1 }));
        Assert.Equal(new int[] { -2147483648 }, add.ToArray());
        var sub = (Tensor<int>)Apply(OpType.Sub, DenseTensor<int>.OfValues(new int[] { -2147483648 }), DenseTensor<int>.OfValues(new int[] { 1 }));
        Assert.Equal(new int[] { 2147483647 }, sub.ToArray());
        var mul = (Tensor<int>)Apply(OpType.Mul, DenseTensor<int>.OfValues(new int[] { 1000000 }), DenseTensor<int>.OfValues(new int[] { 1000000 }));
        Assert.Equal(new int[] { -727379968 }, mul.ToArray());
        var add64 = (Tensor<long>)Apply(OpType.Add, DenseTensor<long>.OfValues(new long[] { 9223372036854775807L }), DenseTensor<long>.OfValues(new long[] { 1L }));
        Assert.Equal(new long[] { -9223372036854775808L }, add64.ToArray());
        var mul64 = (Tensor<long>)Apply(OpType.Mul, DenseTensor<long>.OfValues(new long[] { 3037000500L }), DenseTensor<long>.OfValues(new long[] { 3037000500L }));
        Assert.Equal(new long[] { -9223372036709301616L }, mul64.ToArray());
    }

    static void AssertDivFails(ComputationalGraph g, Dictionary<string, ITensor> inputs, Type cause)
    {
        Assert.False(g.Execute(inputs, false));
        Assert.Equal("n", g.LastFailedNodeName);
        Assert.Equal(OpType.Div, g.LastFailedNodeOp);
        Assert.False(string.IsNullOrEmpty(g.LastErrorMessage));
        Assert.IsType(cause, g.LastErrorCause);
    }

    [Fact]
    public void IntegerDivByZero_FailsCleanlyWithNodeIdentity()
    {
        var zero32 = DenseTensor<int>.OfValues(new int[] { 0 });
        var one32 = DenseTensor<int>.OfValues(new int[] { 1 });
        AssertDivFails(BinaryGraph(OpType.Div, one32, zero32, DenseTensor<int>.OfShape(1)), Pair(one32, zero32), typeof(DivideByZeroException));
        var zero64 = DenseTensor<long>.OfValues(new long[] { 0L });
        var one64 = DenseTensor<long>.OfValues(new long[] { 1L });
        AssertDivFails(BinaryGraph(OpType.Div, one64, zero64, DenseTensor<long>.OfShape(1)), Pair(one64, zero64), typeof(DivideByZeroException));
    }

    [Fact]
    public void NegMin_WrapsToMin()
    {
        // ORT 1.29: Neg(int32_min)=int32_min, Neg(int64_min)=int64_min (wraps, no fault).
        foreach (var opts in new ExecutionOptions[] { ExecutionOptions.Default, ExecutionOptions.Scalar })
        {
            var n32 = CPU.Neg(DenseTensor<int>.OfValues(new int[] { -2147483648, -1, 0, 1 }), opts);
            Assert.Equal(OpStatus.Success, n32.Status);
            Assert.Equal(new int[] { -2147483648, 1, 0, -1 }, ((Tensor<int>)n32.Outputs![0]).ToArray());
            var n64 = CPU.Neg(DenseTensor<long>.OfValues(new long[] { -9223372036854775808L, -1L, 0L, 1L }), opts);
            Assert.Equal(OpStatus.Success, n64.Status);
            Assert.Equal(new long[] { -9223372036854775808L, 1L, 0L, -1L }, ((Tensor<long>)n64.Outputs![0]).ToArray());
        }
    }

    [Fact]
    public void AbsMin_WrapsToMin()
    {
        // ORT 1.29: Abs(int32_min)=int32_min, Abs(int64_min)=int64_min (wraps, no throw).
        var a32 = CPU.Abs(DenseTensor<int>.OfValues(new int[] { -2147483648, -1, 0, 1 }), null);
        Assert.Equal(OpStatus.Success, a32.Status);
        Assert.Equal(new int[] { -2147483648, 1, 0, 1 }, ((Tensor<int>)a32.Outputs![0]).ToArray());
        var a64 = CPU.Abs(DenseTensor<long>.OfValues(new long[] { -9223372036854775808L, -1L, 0L, 1L }), null);
        Assert.Equal(OpStatus.Success, a64.Status);
        Assert.Equal(new long[] { -9223372036854775808L, 1L, 0L, 1L }, ((Tensor<long>)a64.Outputs![0]).ToArray());
    }

    [Fact]
    public void MinDividedByNegOne_FailsCleanlyWithoutFault()
    {
        var x32 = DenseTensor<int>.OfValues(new int[] { -2147483648 });
        var y32 = DenseTensor<int>.OfValues(new int[] { -1 });
        AssertDivFails(BinaryGraph(OpType.Div, x32, y32, DenseTensor<int>.OfShape(1)), Pair(x32, y32), typeof(OverflowException));
        var x64 = DenseTensor<long>.OfValues(new long[] { -9223372036854775808L });
        var y64 = DenseTensor<long>.OfValues(new long[] { -1L });
        AssertDivFails(BinaryGraph(OpType.Div, x64, y64, DenseTensor<long>.OfShape(1)), Pair(x64, y64), typeof(OverflowException));
    }
}
