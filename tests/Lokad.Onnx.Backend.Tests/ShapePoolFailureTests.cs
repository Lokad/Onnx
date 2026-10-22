using System.Runtime.InteropServices;

namespace Lokad.Onnx.Backend.Tests;

public class ShapePoolFailureTests
{
    sealed class FailingInput(Tensor<float> parent, Exception failure)
        : TensorSlice<float>(parent, [SliceIndex.All, SliceIndex.All])
    {
        public override DenseTensor<float> ToDenseTensor() => throw failure;
        public override float this[ReadOnlySpan<int> indices]
        {
            get => throw failure;
            set => base[indices] = value;
        }
    }

    static OpResult Run(string operation, ITensor input, TensorBufferPool? pool) => operation == "transpose"
        ? CPUExecutionProvider.Transpose(input, [1, 0], null, pool)
        : CPUExecutionProvider.Concat([DenseTensor<float>.OfShape(2, 2), input], 0, null, pool);

    static float[] BufferOf(DenseTensor<float> tensor)
    {
        Assert.True(MemoryMarshal.TryGetArray((ReadOnlyMemory<float>)tensor.Buffer, out var segment));
        return Assert.IsType<float[]>(segment.Array);
    }

    [Theory]
    [InlineData("transpose", 4)]
    [InlineData("concat", 8)]
    public void OrdinaryFailuresReturnUnescapedDestinationsAndAllowHealthyReuse(string operation, int length)
    {
        var parent = new DenseTensor<float>(new[] { 1f, -2f, 3f, 4f }, [2, 2]);
        var before = parent.ToArray();
        var failure = new InvalidOperationException("Controlled input conversion/read failure");
        var input = new FailingInput(parent, failure);
        Assert.Same(failure, Record.Exception(() => Run(operation, input, null)));
        var pool = new TensorBufferPool();
        for (int i = 0; i < 4; i++)
        {
            Assert.Same(failure, Record.Exception(() => Run(operation, input, pool)));
            Assert.Equal(before, parent.ToArray());
        }
        Assert.Equal(1, pool.AllocatedNew);
        Assert.Equal(3, pool.Reused);
        Assert.Equal(4, pool.Returned);
        Assert.Equal(length * sizeof(float), pool.PeakOutstandingBytes);

        var returned = pool.Rent<float>(length);
        Array.Fill(returned, float.NaN);
        pool.Return(returned);
        var control = (Tensor<float>)Run(operation, parent, null).Outputs[0];
        var output = Assert.IsType<DenseTensor<float>>(Run(operation, parent, pool).Outputs[0]);
        Assert.Same(returned, BufferOf(output));
        Assert.Equal(control.Dimensions.ToArray(), output.Dimensions.ToArray());
        Assert.Equal(control.ToArray(), output.ToArray());
        Assert.Equal(before, parent.ToArray());
        Assert.Equal(1, pool.AllocatedNew);
    }

    [Theory]
    [InlineData("transpose", 4)]
    [InlineData("concat", 8)]
    public void SuccessfulDestinationsRemainLeasedAndCannotBeOverwritten(string operation, int length)
    {
        var input = new DenseTensor<float>(new[] { 1f, 2f, 3f, 4f }, [2, 2]);
        var pool = new TensorBufferPool();
        var output = Assert.IsType<DenseTensor<float>>(Run(operation, input, pool).Outputs[0]);
        var storage = BufferOf(output);
        var values = output.ToArray();
        Assert.True(pool.IsOwned(storage));
        Assert.Equal(0, pool.Returned);
        var later = pool.Rent<float>(length);
        Assert.NotSame(storage, later);
        Array.Fill(later, 99f);
        Assert.Equal(values, output.ToArray());
        pool.Return(later);
        pool.Return(storage);
    }

    [Theory]
    [InlineData("transpose", false)]
    [InlineData("transpose", true)]
    [InlineData("concat", false)]
    [InlineData("concat", true)]
    public void ControlledFatalFailuresPropagateWithoutRecovery(string operation, bool stackExhaustion)
    {
        // Controlled exception objects, not actual memory/stack exhaustion.
        Exception failure = stackExhaustion ? new StackOverflowException("controlled") : new OutOfMemoryException("controlled");
        var pool = new TensorBufferPool();
        var input = new FailingInput(DenseTensor<float>.OfShape(2, 2), failure);
        Assert.Same(failure, Record.Exception(() => Run(operation, input, pool)));
        Assert.Equal(0, pool.Returned);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void InvalidTransposePermutationDoesNotRent(int invalid)
    {
        int[] perm = invalid == 0 ? [0] : invalid == 1 ? [0, 0] : [0, 2];
        var input = DenseTensor<float>.OfShape(2, 2);
        var expected = Assert.Throws<ArgumentException>(() => CPUExecutionProvider.Transpose(input, perm, null, null));
        var pool = new TensorBufferPool();
        var actual = Assert.Throws<ArgumentException>(() => CPUExecutionProvider.Transpose(input, perm, null, pool));
        Assert.Equal(expected.Message, actual.Message);
        Assert.Equal(expected.ParamName, actual.ParamName);
        Assert.Equal(0, pool.AllocatedNew);
    }

    [Fact]
    public void SplitMaterializesBeforeRentingAnyOutput()
    {
        var failure = new InvalidOperationException("Controlled split conversion failure");
        var input = new FailingInput(DenseTensor<float>.OfShape(2, 2), failure);
        var pool = new TensorBufferPool();
        Assert.Same(failure, Record.Exception(() => CPUExecutionProvider.Split(input, null, 0, [1, 1], null, null, 2, pool)));
        Assert.Equal(0, pool.AllocatedNew);
    }
}
