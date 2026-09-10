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

    [Fact]
    public void UnsignedOverflow_WrapsModuloWidth()
    {
        // ORT 1.29: uint32/uint64 are schema-valid arithmetic types and wrap
        // modulo their width, exactly like the signed widths.
        var add32 = (Tensor<uint>)Apply(OpType.Add, DenseTensor<uint>.OfValues(new uint[] { 4294967295u }), DenseTensor<uint>.OfValues(new uint[] { 1u }));
        Assert.Equal(new uint[] { 0u }, add32.ToArray());
        var sub32 = (Tensor<uint>)Apply(OpType.Sub, DenseTensor<uint>.OfValues(new uint[] { 0u }), DenseTensor<uint>.OfValues(new uint[] { 1u }));
        Assert.Equal(new uint[] { 4294967295u }, sub32.ToArray());
        var mul32 = (Tensor<uint>)Apply(OpType.Mul, DenseTensor<uint>.OfValues(new uint[] { 65536u }), DenseTensor<uint>.OfValues(new uint[] { 65536u }));
        Assert.Equal(new uint[] { 0u }, mul32.ToArray());
        var div32 = CPU.Div(DenseTensor<uint>.OfValues(new uint[] { 7u, 8u }), DenseTensor<uint>.OfValues(new uint[] { 2u, 3u }), null, null);
        Assert.Equal(OpStatus.Success, div32.Status);
        Assert.Equal(new uint[] { 3u, 2u }, ((Tensor<uint>)div32.Outputs![0]).ToArray());
        var add64 = (Tensor<ulong>)Apply(OpType.Add, DenseTensor<ulong>.OfValues(new ulong[] { 18446744073709551615ul }), DenseTensor<ulong>.OfValues(new ulong[] { 1ul }));
        Assert.Equal(new ulong[] { 0ul }, add64.ToArray());
        var sub64 = (Tensor<ulong>)Apply(OpType.Sub, DenseTensor<ulong>.OfValues(new ulong[] { 0ul }), DenseTensor<ulong>.OfValues(new ulong[] { 1ul }));
        Assert.Equal(new ulong[] { 18446744073709551615ul }, sub64.ToArray());
        var mul64 = (Tensor<ulong>)Apply(OpType.Mul, DenseTensor<ulong>.OfValues(new ulong[] { 4294967296ul }), DenseTensor<ulong>.OfValues(new ulong[] { 4294967296ul }));
        Assert.Equal(new ulong[] { 0ul }, mul64.ToArray());
        var div64 = CPU.Div(DenseTensor<ulong>.OfValues(new ulong[] { 7ul }), DenseTensor<ulong>.OfValues(new ulong[] { 2ul }), null, null);
        Assert.Equal(OpStatus.Success, div64.Status);
        Assert.Equal(new ulong[] { 3ul }, ((Tensor<ulong>)div64.Outputs![0]).ToArray());
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

    static DenseTensor<int> Scalar32(int value)
    {
        var s = DenseTensor<int>.OfShape();
        s.SetValue(0, value);
        return s;
    }

    static DenseTensor<long> Scalar64(long value)
    {
        var s = DenseTensor<long>.OfShape();
        s.SetValue(0, value);
        return s;
    }

    [Fact]
    public void ScalarNegAbsMin_WrapsToMin()
    {
        // ORT 1.29 rank-0: Neg(int32_min)=int32_min, Abs(int32_min)=int32_min (wraps, no fault).
        var neg = CPU.Neg(Scalar32(-2147483648), null);
        Assert.Equal(OpStatus.Success, neg.Status);
        Assert.Equal(new int[] { -2147483648 }, ((Tensor<int>)neg.Outputs![0]).ToArray());
        var abs = CPU.Abs(Scalar32(-2147483648), null);
        Assert.Equal(OpStatus.Success, abs.Status);
        Assert.Equal(new int[] { -2147483648 }, ((Tensor<int>)abs.Outputs![0]).ToArray());
        // ORT 1.29 rank-0: Neg(int64_min)=int64_min, Abs(int64_min)=int64_min.
        var neg64 = CPU.Neg(Scalar64(-9223372036854775808L), null);
        Assert.Equal(OpStatus.Success, neg64.Status);
        Assert.Equal(new long[] { -9223372036854775808L }, ((Tensor<long>)neg64.Outputs![0]).ToArray());
        var abs64 = CPU.Abs(Scalar64(-9223372036854775808L), null);
        Assert.Equal(OpStatus.Success, abs64.Status);
        Assert.Equal(new long[] { -9223372036854775808L }, ((Tensor<long>)abs64.Outputs![0]).ToArray());
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
