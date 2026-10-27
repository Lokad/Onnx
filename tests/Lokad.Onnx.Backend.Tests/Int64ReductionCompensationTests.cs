using System;
using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class Int64ReductionCompensationTests
{
    // Exact outputs checked against ORT 1.29 CPU, one thread, optimizations off.
    // Its serial Int64 aggregators use compensated double accumulation, not
    // exact integer arithmetic. Keep conversion rounding and final saturation.
    public static IEnumerable<object[]> ReferenceCases()
    {
        yield return ["ReduceSum", new long[] { 1L << 53, 1, -(1L << 53) }, 1L];
        yield return ["ReduceSum", CancellationRow(1), 1L];
        yield return ["ReduceSum", new long[] { (1L << 53) + 1 }, 1L << 53];
        yield return ["ReduceSum", new long[] { (1L << 53) + 1, -(1L << 53) }, 0L];
        yield return ["ReduceSum", new[] { long.MaxValue, long.MaxValue, long.MinValue, long.MinValue }, 0L];
        yield return ["ReduceSum", new[] { long.MaxValue, long.MaxValue }, long.MaxValue];
        yield return ["ReduceSum", new[] { long.MinValue, long.MinValue }, long.MinValue];
        yield return ["ReduceMean", new long[] { 1L << 53, 9, -(1L << 53) }, 3L];
        yield return ["ReduceMean", CancellationRow(65), 1L];
        yield return ["ReduceMean", new long[] { (1L << 53) + 1, 1 }, 1L << 52];
        yield return ["ReduceMean", new long[] { -1, -2 }, -1L];
        yield return ["ReduceMean", new[] { long.MaxValue, long.MaxValue }, long.MaxValue];
        yield return ["ReduceSumSquare", new long[] { 1L << 27, 1, 1, 1, 1 }, (1L << 54) + 4];
        yield return ["ReduceSumSquare", new long[] { 1L << 26, 1, 1, 1, 1 }, (1L << 52) + 4];
        yield return ["ReduceSumSquare", new[] { long.MaxValue, long.MinValue }, long.MaxValue];
    }

    [Theory]
    [MemberData(nameof(ReferenceCases))]
    public void SerialReferenceResults(string op, long[] input, long expected)
        => Check(op, DenseTensor<long>.OfValues(input), [0], false, false, [], [expected]);

    public static IEnumerable<object[]> LayoutCases()
    {
        foreach (string op in new[] { "ReduceSum", "ReduceMean", "ReduceSumSquare" })
        foreach (int layout in Enumerable.Range(0, 5))
        foreach (bool keep in new[] { false, true })
            yield return [op, layout, keep];
    }

    [Theory]
    [MemberData(nameof(LayoutCases))]
    // Apply the serial policy consistently to views. ORT 1.29's row Mean fast
    // path drops compensation, so those cases intentionally differ from ORT.
    public void PartialReductionPreservesLogicalLayout(string op, int layout, bool keep)
    {
        var row = op == "ReduceSumSquare" ? new long[] { 1L << 27, 1, 1, 1, 1 }
            : CancellationRow(op == "ReduceMean" ? 65 : 1);
        long first = op == "ReduceSumSquare" ? (1L << 54) + 4 : 1;
        int n = row.Length;
        Tensor<long> input;
        if (layout == 2)
        {
            var padded = DenseTensor<long>.OfShape(2, n + 1);
            padded[0, n] = padded[1, n] = long.MaxValue;
            input = padded.Slice(SliceIndex.All, new SliceIndex(0, n));
        }
        else input = new DenseTensor<long>(new[] { layout == 4 ? 1 : 2, n }, layout == 1);
        for (int r = 0; r < (layout == 4 ? 1 : 2); r++)
        for (int c = 0; c < n; c++) input[r, c] = r == 0 ? row[c] : -row[c];
        if (layout == 3) input = Tensor<long>.Transpose(input, [1, 0]);
        if (layout == 4) input = input.BroadcastDim(0, 2);
        int axis = layout == 3 ? 0 : 1;
        var shape = !keep ? new[] { 2 } : axis == 0 ? new[] { 1, 2 } : new[] { 2, 1 };
        Check(op, input, [axis], keep, false, shape,
            [first, layout == 4 || op == "ReduceSumSquare" ? first : -first]);
    }

    [Theory]
    [InlineData("ReduceSum")]
    [InlineData("ReduceMean")]
    [InlineData("ReduceSumSquare")]
    public void AxisNormalizationDoesNotChangeAccumulation(string op)
    {
        var row = op == "ReduceSumSquare" ? new long[] { 1L << 27, 1, 1, 1, 1 }
            : CancellationRow(op == "ReduceMean" ? 65 : 1);
        var input = new DenseTensor<long>(row, new[] { 1, row.Length });
        long expected = op == "ReduceSumSquare" ? (1L << 54) + 4 : 1;
        foreach (var axes in new long[][] { [0, 1], [1, 0], [-1, -2, -1] })
        foreach (bool keep in new[] { false, true })
            Check(op, input, axes, keep, false, keep ? [1, 1] : [], [expected]);
    }

    [Theory]
    [InlineData("ReduceSum")]
    [InlineData("ReduceMean")]
    [InlineData("ReduceSumSquare")]
    public void EmptyReductionAndNoopKeepTheirSemantics(string op)
    {
        foreach (bool keep in new[] { false, true })
        {
            Check(op, DenseTensor<long>.OfShape(2, 0), [1], keep, false, keep ? [2, 1] : [2], [0, 0]);
            var input = DenseTensor<long>.OfValues(new long[] { 3, -4, 1L << 32 });
            long[] expected = op == "ReduceSumSquare" ? [9, 16, long.MaxValue] : input.ToArray();
            Check(op, input, [], keep, true, [3], expected);
            Check(op, input, null, keep, true, [3], expected);
        }
    }

    static long[] CancellationRow(long middle) => Enumerable.Repeat(1L << 48, 32)
        .Append(middle).Concat(Enumerable.Repeat(-(1L << 48), 32)).ToArray();

    static void Check(string op, Tensor<long> input, long[]? axes, bool keep, bool noop, int[] shape, long[] expected)
    {
        var original = input.ToArray();
        var axisTensor = axes is null ? null : DenseTensor<long>.OfValues(axes);
        var originalAxes = axisTensor?.ToArray();
        var model = new OnnxModel
        {
            Opset = { [""] = 18 },
            Inputs = { new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Int64, Dims = input.Dimensions.ToArray() } },
            Outputs = { new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Int64, Dims = shape } },
            Nodes = { new OnnxNode { OpType = op, Inputs = axes is null ? ["x"] : ["x", "axes"], Outputs = ["y"],
                Attributes = new() { ["keepdims"] = keep ? 1L : 0L, ["noop_with_empty_axes"] = noop ? 1L : 0L } } }
        };
        if (axes is not null) model.Initializers.Add(new OnnxTensor
            { Name = "axes", ElementType = TensorElementType.Int64, Dims = [axes.Length], Data = axes });
        var graph = Model.Load(model);
        foreach (var options in new[] { ExecutionOptions.Default, ExecutionOptions.Scalar, ExecutionOptions.Simd })
        {
            var result = op switch
            {
                "ReduceSum" => CPUExecutionProvider.ReduceSum(input, axisTensor, keep ? 1 : 0, noop ? 1 : 0, options),
                "ReduceMean" => CPUExecutionProvider.ReduceMean(input, axisTensor, keep ? 1 : 0, noop ? 1 : 0, options),
                _ => CPUExecutionProvider.ReduceSumSquare(input, axisTensor, keep ? 1 : 0, noop ? 1 : 0, options)
            };
            Assert.Equal(OpStatus.Success, result.Status);
            CheckOutput((Tensor<long>)result.Outputs[0]);
            Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true, ExecutionProvider.CPU, options), graph.LastErrorMessage);
            CheckOutput((Tensor<long>)graph.Outputs["y"]);
            Assert.Equal(original, input.ToArray());
            if (axisTensor is not null) Assert.Equal(originalAxes, axisTensor.ToArray());
        }

        void CheckOutput(Tensor<long> output)
        {
            Assert.Equal(shape, output.Dimensions.ToArray());
            Assert.Equal(expected, output.ToArray());
            if (noop && (axes is null || axes.Length == 0) && output.Length > 0)
            {
                Assert.NotSame(input, output);
                output.SetValue(0, 123);
                Assert.Equal(original, input.ToArray());
            }
        }
    }
}
