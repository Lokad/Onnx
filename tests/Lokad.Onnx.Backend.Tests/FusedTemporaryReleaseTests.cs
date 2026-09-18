namespace Lokad.Onnx.Backend.Tests;

public class FusedTemporaryReleaseTests
{
    static DenseTensor<float> Input(int[] dims, int seed)
    {
        int length = dims.Aggregate(1, (a,b) => a*b);
        var random = new Random(seed);
        return new DenseTensor<float>(Enumerable.Range(0,length).Select(_ => (float)random.NextDouble() - .5f).ToArray(), dims);
    }

    static int[] Bits(ITensor tensor) => ((Tensor<float>)tensor).ToArray().Select(BitConverter.SingleToInt32Bits).ToArray();

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void BatchedProductIsReclaimedWithoutChangingArithmeticOrReturnedValues(int mode)
    {
        var options = mode == 0 ? ExecutionOptions.Scalar : mode == 1 ? ExecutionOptions.Simd : ExecutionOptions.Intrinsics;
        var a = Input(new[] { 1,2,3,4 }, 5);
        var b = Input(new[] { 1,2,4,5 }, 7);
        var divisor = DenseTensor<float>.Scalar(5.656854f);
        var originalA = Bits(a);
        var originalB = Bits(b);
        var baseline = CPUExecutionProvider.ScaledMatMulTrailing(a,b,divisor,options,null);
        Assert.Equal(OpStatus.Success, baseline.Status);
        var pool = new TensorBufferPool { ReleaseFusedTemporaries = true };
        var poison = pool.Rent<float>(30);
        Array.Fill(poison, float.NaN);
        pool.Return(poison);
        var first = CPUExecutionProvider.ScaledMatMulTrailing(a,b,divisor,options,pool);
        Assert.Equal(OpStatus.Success, first.Status);
        Assert.Equal(Bits(baseline.Outputs[0]), Bits(first.Outputs[0]));
        Assert.Equal(2, pool.Returned); // Seed + private product, never the result.
        var snapshot = Bits(first.Outputs[0]);
        var reused = pool.Rent<float>(30);
        Assert.Same(poison, reused);
        Array.Fill(reused, -12345f);
        Assert.Equal(snapshot, Bits(first.Outputs[0]));
        pool.Return(reused);
        var second = CPUExecutionProvider.ScaledMatMulTrailing(a,b,divisor,options,pool);
        Assert.Equal(snapshot, Bits(second.Outputs[0]));
        Assert.Equal(snapshot, Bits(first.Outputs[0]));
        Assert.Equal(originalA, Bits(a));
        Assert.Equal(originalB, Bits(b));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RepeatedUnboundProductsHaveBoundedStorageOnlyWhenEnabled(bool enabled)
    {
        var a = Input(new[] { 1,2,3,4 }, 5);
        var b = Input(new[] { 1,2,4,5 }, 7);
        var pool = new TensorBufferPool { ReleaseFusedTemporaries = enabled };
        for (int i = 0; i < 12; i++)
        {
            var r = CPUExecutionProvider.ScaledMatMulTrailing(a,b,DenseTensor<float>.Scalar(4f),ExecutionOptions.Scalar,pool);
            Assert.Equal(OpStatus.Success,r.Status);
            pool.Return(((Tensor<float>)r.Outputs[0]).OwnedBufferArray()!);
        }
        Assert.Equal(enabled ? 2 : 13, pool.AllocatedNew);
        Assert.Equal(enabled ? 24 : 12, pool.Returned);
        Assert.Equal(enabled ? 240 : 1560, pool.PeakOutstandingBytes);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void DivisionFailureReleasesProductAndPreservesTheError(int failure)
    {
        var a = Input(new[] { 1,2,3,4 }, 5);
        var b = Input(new[] { 1,2,4,5 }, 7);
        ITensor divisor = failure == 0 ? DenseTensor<int>.Scalar(2) : Input(new[] { 7 }, 9);
        var baseline = CPUExecutionProvider.ScaledMatMulTrailing(a,b,divisor,ExecutionOptions.Scalar,null);
        var pool = new TensorBufferPool { ReleaseFusedTemporaries = true };
        var actual = CPUExecutionProvider.ScaledMatMulTrailing(a,b,divisor,ExecutionOptions.Scalar,pool);
        Assert.Equal(OpStatus.Failure, actual.Status);
        Assert.Equal(baseline.Message, actual.Message);
        Assert.Equal(1, pool.Returned);
        Assert.Equal(1, pool.AllocatedNew);
        _ = pool.Rent<float>(30);
        Assert.Equal(1, pool.Reused);
    }

    [Fact]
    public void ExceptionalDivisorsAndVectorScalarOutputsRemainBitwiseIdentical()
    {
        var a = new DenseTensor<float>(new[] { 0f, -0f, 1f, float.PositiveInfinity }, new[] { 4 });
        var b = new DenseTensor<float>(new[] { 2f, -2f, 3f, 0f }, new[] { 4 });
        foreach (float d in new[] { 0f, -0f, 1f, float.NaN, float.PositiveInfinity })
        {
            var divisor = DenseTensor<float>.Scalar(d);
            var baseline = CPUExecutionProvider.ScaledMatMulTrailing(a,b,divisor,ExecutionOptions.Scalar,null);
            var pool = new TensorBufferPool { ReleaseFusedTemporaries = true };
            var result = CPUExecutionProvider.ScaledMatMulTrailing(a,b,divisor,ExecutionOptions.Scalar,pool);
            Assert.Equal(OpStatus.Success, result.Status);
            Assert.Equal(Bits(baseline.Outputs[0]), Bits(result.Outputs[0]));
            Assert.Equal(0, result.Outputs[0].Rank);
            Assert.Equal(1, pool.Returned);
            Array.Fill(pool.Rent<float>(1), 999f);
            Assert.Equal(Bits(baseline.Outputs[0]), Bits(result.Outputs[0]));
        }
    }

    [Fact]
    public void EmptyProductAndExpandedDivisionDoNotReturnCallerStorage()
    {
        var a = new DenseTensor<float>(Array.Empty<float>(), new[] { 2,0 });
        var b = new DenseTensor<float>(Array.Empty<float>(), new[] { 0,3 });
        var divisor = Input(new[] { 4,2,3 }, 7);
        var expected = CPUExecutionProvider.ScaledMatMulTrailing(a,b,divisor,ExecutionOptions.Scalar,null);
        var pool = new TensorBufferPool { ReleaseFusedTemporaries = true };
        var actual = CPUExecutionProvider.ScaledMatMulTrailing(a,b,divisor,ExecutionOptions.Scalar,pool);
        Assert.Equal(OpStatus.Success, actual.Status);
        Assert.Equal(new[] { 4,2,3 }, actual.Outputs[0].Dims);
        Assert.Equal(Bits(expected.Outputs[0]), Bits(actual.Outputs[0]));
        Assert.Equal(1, pool.Returned);
        Array.Fill(pool.Rent<float>(6), 999f);
        Assert.Equal(Bits(expected.Outputs[0]), Bits(actual.Outputs[0]));
    }

    [Fact]
    public void NonFloatPathDoesNotAdoptAnUnpooledProduct()
    {
        var a = new DenseTensor<double>(new[] { 2.0,3.0 }, new[] { 1,2 });
        var b = new DenseTensor<double>(new[] { 4.0,5.0 }, new[] { 2,1 });
        var pool = new TensorBufferPool { ReleaseFusedTemporaries = true };
        var result = CPUExecutionProvider.ScaledMatMulTrailing(a,b,DenseTensor<double>.Scalar(2),ExecutionOptions.Scalar,pool);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(11.5, ((Tensor<double>)result.Outputs[0]).GetValue(0));
        Assert.Equal(0,pool.Returned);
        Assert.Equal(0,pool.AllocatedNew);
    }

    [Fact]
    public void PrivateProductCanCrossCallsWhileHeldOutputsRemainReadable()
    {
        var cache = new ReleasedBufferCache(4096,16);
        var a = Input(new[] { 1,2,3,4 },5);
        var b = Input(new[] { 1,2,4,5 },7);
        var held = new List<(ITensor Tensor, int[] Bits)>();
        for (int run = 0; run < 20; run++)
        {
            var pool = new TensorBufferPool(cache) { ReleaseFusedTemporaries = true };
            var result = CPUExecutionProvider.ScaledMatMulTrailing(a,b,DenseTensor<float>.Scalar(run + 1f),ExecutionOptions.Scalar,pool);
            Assert.Equal(OpStatus.Success,result.Status);
            held.Add((result.Outputs[0],Bits(result.Outputs[0])));
            pool.RetainReleasedBuffers();
            Assert.Equal(1,cache.Count);
            Assert.Equal(120,cache.Bytes);
            Assert.Equal(run == 0 ? 2 : 1,pool.AllocatedNew);
            foreach (var old in held) Assert.Equal(old.Bits,Bits(old.Tensor));
        }
    }

    [Fact]
    public void ZeroElementProductCanBeReleasedAndReused()
    {
        var pool = new TensorBufferPool { ReleaseFusedTemporaries = true };
        var a = new DenseTensor<float>(Array.Empty<float>(),new[] { 0,4 });
        var b = Input(new[] { 4,3 },7);
        var result = CPUExecutionProvider.ScaledMatMulTrailing(a,b,DenseTensor<float>.Scalar(2),ExecutionOptions.Scalar,pool);
        Assert.Equal(OpStatus.Success,result.Status);
        Assert.Equal(new[] { 0,3 },result.Outputs[0].Dims);
        Assert.Equal(1,pool.Returned);
        Assert.Empty(pool.Rent<float>(0));
        Assert.Equal(1,pool.Reused);
    }
}
