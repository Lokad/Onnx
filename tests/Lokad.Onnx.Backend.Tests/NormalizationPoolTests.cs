using System.Runtime.InteropServices;

namespace Lokad.Onnx.Backend.Tests;

public class NormalizationPoolTests
{
    [Theory]
    [InlineData("layer", 0)]
    [InlineData("layer", 1)]
    [InlineData("layer", 2)]
    [InlineData("rotary", 0)]
    [InlineData("rotary", 1)]
    [InlineData("rotary", 2)]
    [InlineData("rotary", 3)]
    public void RejectedCallsReturnUnescapedStorageForReuse(string kind, int invalid)
    {
        var pool = new TensorBufferPool();
        var expected = Assert.Throws<ArgumentException>(() => Run(kind, invalid, null));
        for (int i = 0; i < 4; i++)
        {
            var actual = Assert.Throws<ArgumentException>(() => Run(kind, invalid, pool));
            Assert.Equal(expected.Message, actual.Message);
            Assert.Equal(expected.ParamName, actual.ParamName);
        }
        Assert.Equal(1, pool.AllocatedNew);
        Assert.Equal(3, pool.Reused);
        Assert.Equal(4, pool.Returned);
        Assert.Equal(8 * sizeof(float), pool.PeakOutstandingBytes);
        var reused = pool.Rent<float>(8);
        Array.Fill(reused, float.NaN);
        pool.Return(reused);
        var control = (Tensor<float>)Run(kind, -1, null).Outputs![0];
        var result = (Tensor<float>)Run(kind, -1, pool).Outputs![0];
        Assert.Equal(control.ToArray(), result.ToArray());
        Assert.Equal(1, pool.AllocatedNew);
    }

    [Theory]
    [InlineData("layer")]
    [InlineData("rotary")]
    public void SuccessfulDestinationsStayCheckedOutAndCannotBeOverwrittenByAnotherRent(string kind)
    {
        var pool = new TensorBufferPool();
        var result = (DenseTensor<float>)Run(kind, -1, pool).Outputs![0];
        var expected = result.ToArray();
        Assert.True(MemoryMarshal.TryGetArray((ReadOnlyMemory<float>)result.Buffer, out var storage));
        Assert.True(pool.IsOwned(storage.Array!));
        Assert.Equal(0, pool.Returned);
        var later = pool.Rent<float>(8);
        Assert.NotSame(storage.Array, later);
        Array.Fill(later, 99f);
        Assert.Equal(expected, result.ToArray());
        pool.Return(later);
    }

    static OpResult Run(string kind, int invalid, TensorBufferPool? pool)
    {
        var input = new DenseTensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, [2, 4]);
        if (kind == "layer")
        {
            var scale = DenseTensor<float>.OfValues(Enumerable.Repeat(1f, invalid == 0 ? 3 : 4).ToArray());
            var bias = invalid == 1 ? DenseTensor<float>.OfShape(3) : null;
            return CPUExecutionProvider.LayerNormalization(input, scale, bias, invalid == 2 ? 2 : -1, null, null, 1, null, pool);
        }
        var cos = new DenseTensor<float>(new[] { 1f }, Array.Empty<int>());
        var sin = new DenseTensor<float>(new[] { 0f }, Array.Empty<int>());
        return CPUExecutionProvider.RotaryEmbedding(input, cos, invalid == 3 ? DenseTensor<float>.OfShape(2) : sin,
            invalid == 0 ? 0 : 2, invalid == 1 ? 2 : -1, invalid == 2 ? 0 : -1, null, pool);
    }
}
