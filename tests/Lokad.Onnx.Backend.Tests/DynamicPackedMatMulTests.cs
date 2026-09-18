using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

public class DynamicPackedMatMulTests
{
    static void EqualBits(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(actual[i]));
    }

    // Preserved AVX2 consumers, including the original separate odd-row fixup.
    // This reference never uses either opt-in AVX-512 dispatch.
    static unsafe float[] Reference(float[] a, float[] b, int m, int n, int k)
    {
        var packed = new float[n * k];
        var output = new float[m * k];
        fixed (float* ap = a, bp = b, pp = packed, cp = output)
        {
            MathOps.PackPanelsB(n, k, bp, pp);
            if (m % 3 == 0)
                MathOps.mm_unsafe_vectorized_intrinsics_3x4packed(m, n, k, ap, pp, cp);
            else
            {
                int blocked = m - m % 2;
                MathOps.mm_unsafe_vectorized_intrinsics_2x4packed_bump(blocked, n, k, ap, pp, cp);
                if (blocked != m)
                    MathOps.mm_unsafe_vectorized_intrinsics(1, n, k, ap + blocked * n, bp, cp + blocked * k);
            }
        }
        return output;
    }

    [SkippableTheory]
    [InlineData(63, 64, 32)]
    [InlineData(64, 63, 32)]
    [InlineData(64, 64, 32)]
    [InlineData(65, 65, 32)]
    [InlineData(66, 64, 64)]
    [InlineData(67, 65, 64)]
    [InlineData(128, 128, 32)]
    [InlineData(512, 512, 32)]
    [InlineData(128, 32, 128)]
    [InlineData(512, 32, 512)]
    [InlineData(64, 65, 33)]
    public void DynamicProductsMatchPreservedKernelsAndRepackMutatedInputs(int m, int n, int k)
    {
        Skip.IfNot(Fma.IsSupported, "The independent AVX2 reference requires FMA.");
        var random = new Random(713 + m + n + k);
        var a = Enumerable.Range(0, m * n).Select(_ => random.NextSingle() * 2 - 1).ToArray();
        var b = Enumerable.Range(0, n * k).Select(_ => random.NextSingle() * 2 - 1).ToArray();
        var originalA = (float[])a.Clone();
        var x = new DenseTensor<float>(a, new[] { m, n });
        var y = new DenseTensor<float>(b, new[] { n, k });
        var held = Tensor<float>.MatMul2D(x, y, TensorExecutionOptions.Intrinsics);
        var heldValues = Reference(a, b, m, n, k);
        EqualBits(heldValues, held.ToArray());

        // Caller-owned destination has guards and poisoned contents before every call.
        const float guard = -12345.5f;
        var backing = Enumerable.Repeat(guard, m * k + 6).ToArray();
        var destination = new DenseTensor<float>(backing.AsMemory(3, m * k), new[] { m, k });
        var scratch = new ScratchAccountant();
        var options = TensorExecutionOptions.Intrinsics with { ScratchReporter = scratch };
        for (int rep = 0; rep < 2; rep++)
        {
            b[13] += 0.25f;
            var beforeB = (float[])b.Clone();
            destination.Buffer.Span.Fill(float.NaN);
            Assert.Same(destination, Tensor<float>.MatMul2D(x, y, destination, options));
            EqualBits(Reference(a, b, m, n, k), destination.ToArray());
            EqualBits(originalA, a);
            EqualBits(beforeB, b);
            EqualBits(heldValues, held.ToArray());
            Assert.All(backing.Take(3).Concat(backing.TakeLast(3)), value => Assert.Equal(guard, value));
        }
        Assert.Equal(2L * n * k * sizeof(float), scratch.TotalScratchBytes);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BatchedBroadcastProductsPreserveOutputsAcrossMutationResetAndFailure(bool explicitContext)
    {
        const int batches = 2, m = 65, n = 128, k = 32;
        var model = new OnnxModel { Name = "dynamic-batched-product" };
        model.Inputs.Add(new OnnxValueInfo { Name = "a", ElementType = TensorElementType.Float, Dims = new[] { batches, m, n } });
        model.Inputs.Add(new OnnxValueInfo { Name = "b", ElementType = TensorElementType.Float, Dims = new[] { 1, n, k } });
        model.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { batches, m, k } });
        model.Nodes.Add(new OnnxNode { Name = "product", OpType = "MatMul", Inputs = new[] { "a", "b" }, Outputs = new[] { "y" } });
        var graph = Model.Load(model) ?? throw new InvalidOperationException("Model did not load.");
        ComputationalGraph execution = explicitContext ? graph.CreateExecution(null) : graph;
        // Quarter-integers keep the scalar reference exact on every hardware mode.
        var a = Enumerable.Range(0, batches * m * n).Select(i => (i % 11 - 5) * 0.25f).ToArray();
        var b = Enumerable.Range(0, n * k).Select(i => (i % 7 - 3) * 0.25f).ToArray();
        var originalA = (float[])a.Clone();
        var x = new DenseTensor<float>(a, new[] { batches, m, n });
        var y = new DenseTensor<float>(b, new[] { 1, n, k });
        var feed = new Dictionary<string, ITensor> { ["a"] = x, ["b"] = y };
        var retained = new List<(Tensor<float> Tensor, float[] Values)>();
        for (int rep = 0; rep < 3; rep++)
        {
            b[17] += 0.25f;
            var beforeB = (float[])b.Clone();
            var expected = Tensor<float>.MatMul(x, y, TensorExecutionOptions.Scalar).ToArray();
            Assert.True(execution.Execute(feed, false), execution.LastErrorMessage);
            var result = Assert.IsAssignableFrom<Tensor<float>>(execution.Outputs["y"]);
            EqualBits(expected, result.ToArray());
            retained.Add((result, expected));
            execution.Reset();
            Assert.False(execution.Execute(new Dictionary<string, ITensor>(), false));
            foreach (var item in retained) EqualBits(item.Values, item.Tensor.ToArray());
            EqualBits(originalA, a);
            EqualBits(beforeB, b);
        }
    }
}
