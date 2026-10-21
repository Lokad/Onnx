using System.Numerics;

namespace Lokad.Onnx.Tensors.Tests;

public class ElementwiseOverlapTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (string operation in new[] { "unary", "binary", "vectorUnary", "vectorBinary", "broadcast", "delegate" })
        foreach (bool simd in new[] { false, true })
        foreach (int shift in new[] { -1, 0, 1 })
            yield return new object[] { operation, simd, shift };
    }

    static void Run(string operation, Tensor<float> source, Tensor<float> right, Tensor<float> destination, TensorExecutionOptions options)
    {
        switch (operation)
        {
            case "unary": source.Apply(static x => x * 2, destination); break;
            case "binary": source.Apply(static (x, y) => x + y, right, destination); break;
            case "vectorUnary": source.VectorizedApply(static x => x + x, static x => x * 2, destination, options); break;
            case "vectorBinary": source.VectorizedApply(static (x, y) => x + y, static (x, y) => x + y, right, destination, options); break;
            case "broadcast": Tensor<float>.Add(source, right, destination, options); break;
            case "delegate": source.BroadcastApply(right, static (x, y) => x + y, destination, options); break;
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void ShiftedOrExactDenseWindows_ObserveOriginalInputs(string operation, bool simd, int shift)
    {
        var storage = Enumerable.Range(1, 67).Select(x => (float)x).ToArray();
        int sourceStart = shift < 0 ? 1 : 0, destinationStart = shift > 0 ? 1 : 0;
        var source = new DenseTensor<float>(storage.AsMemory(sourceStart, 65), new[] { 65 });
        var destination = new DenseTensor<float>(storage.AsMemory(destinationStart, 65), new[] { 65 });
        var right = DenseTensor<float>.OfValues(Enumerable.Repeat(10f, 65).ToArray());
        var original = source.ToArray();
        Run(operation, source, right, destination, simd ? TensorExecutionOptions.Simd : TensorExecutionOptions.Scalar);
        Assert.Equal(original.Select(x => operation is "unary" or "vectorUnary" ? x * 2 : x + 10), destination.ToArray());
    }

    [Fact]
    public void BroadcastReuseAndNegativeStepOverlap_PreserveOriginalValues()
    {
        foreach (bool useDelegate in new[] { false, true })
        {
            float[] storage = { 1, 2, 0, 0 };
            var source = new DenseTensor<float>(storage.AsMemory(0, 2), new[] { 1, 2 });
            var destination = new DenseTensor<float>(storage, new[] { 2, 2 });
            var ones = DenseTensor<float>.Ones(2, 2);
            if (useDelegate) source.BroadcastApply(ones, static (x, y) => x + y, destination, TensorExecutionOptions.Scalar);
            else Tensor<float>.Add(source, ones, destination, TensorExecutionOptions.Scalar);
            Assert.Equal(new float[] { 2, 3, 2, 3 }, destination.ToArray());
        }
        var parent = DenseTensor<float>.OfValues(new float[] { 1, 2, 3, 4 });
        var reversed = parent.Slice(new SliceIndex(null, null, -1));
        reversed.Apply(static x => x * 2, parent);
        Assert.Equal(new float[] { 8, 6, 4, 2 }, parent.ToArray());
    }

    [Fact]
    public void InPlaceBroadcastViewAndIdenticalBinaryOperands_AreSafe()
    {
        var source = DenseTensor<float>.OfValues(new float[] { 1, 2 }).Reshape(1, 2).BroadcastDim(0, 2);
        source.Apply(static x => x * 2, source);
        Assert.Equal(new float[] { 2, 4, 2, 4 }, source.ToArray());
        var storage = new float[] { 1, 2, 3, 4, 5 };
        var left = new DenseTensor<float>(storage.AsMemory(0, 4), new[] { 4 });
        var destination = new DenseTensor<float>(storage.AsMemory(1, 4), new[] { 4 });
        left.Apply(static (x, y) => x + y, left, destination);
        Assert.Equal(new float[] { 2, 4, 6, 8 }, destination.ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ApplyAndVectorizedApply_PairLogicalPositionsAcrossLayouts(bool simd)
    {
        var source = new DenseTensor<float>(new float[] { 1, 3, 2, 4 }, new[] { 2, 2 }, true);
        var right = new DenseTensor<float>(new float[] { 10, 20, 30, 40 }, new[] { 2, 2 });
        foreach (string operation in new[] { "unary", "binary", "vectorUnary", "vectorBinary" })
        {
            var destination = DenseTensor<float>.OfShape(2, 2);
            Run(operation, source, right, destination, simd ? TensorExecutionOptions.Simd : TensorExecutionOptions.Scalar);
            Assert.Equal(operation is "unary" or "vectorUnary" ? new float[] { 2, 4, 6, 8 } : new float[] { 11, 22, 33, 44 }, destination.ToArray());
        }
    }

    sealed class CustomApply : DenseTensor<float>
    {
        public int Calls;
        public CustomApply() : base(4) { }
        public override void Apply(Func<float, float> operation, Tensor<float> destination)
        {
            Calls++;
            base.Apply(operation, destination);
        }
        public override void Apply(Func<float, float, float> operation, Tensor<float> other, Tensor<float> destination)
        {
            Calls++;
            base.Apply(operation, other, destination);
        }
    }

    [Fact]
    public void VectorFallback_RetainsVirtualApplyDispatch()
    {
        var source = new CustomApply();
        var destination = DenseTensor<float>.OfShape(4);
        source.VectorizedApply(static x => x + x, static x => x * 2, destination, TensorExecutionOptions.Simd);
        source.VectorizedApply(static (x, y) => x + y, static (x, y) => x + y, destination, destination, TensorExecutionOptions.Simd);
        Assert.Equal(2, source.Calls);
    }

    [Fact]
    public void ExactInPlaceAndDisjointDenseWindows_AvoidPayloadSnapshots()
    {
        var storage = new float[131072];
        var source = new DenseTensor<float>(storage.AsMemory(0, 65536), new[] { 65536 });
        var disjoint = new DenseTensor<float>(storage.AsMemory(65536, 65536), new[] { 65536 });
        foreach (var destination in new[] { source, disjoint })
        foreach (bool simd in new[] { false, true })
        {
            void Apply()
            {
                if (simd) source.VectorizedApply(static x => x + Vector<float>.One, static x => x + 1, destination, TensorExecutionOptions.Simd);
                else source.Apply(static x => x + 1, destination);
            }
            Apply();
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 10; i++) Apply();
            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            Assert.True(allocated < 4096, $"Safe dense windows allocated {allocated} bytes.");
        }
    }
}
