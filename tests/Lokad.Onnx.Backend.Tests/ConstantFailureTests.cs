namespace Lokad.Onnx.Backend.Tests;

public class ConstantFailureTests
{
    sealed class ConversionFailure : DenseTensor<float>
    {
        readonly Exception failure;
        public int CloneCalls { get; private set; }

        public ConversionFailure(Exception failure) : base(new float[] { -0f, 2.5f, float.NaN }, new[] { 3 })
            => this.failure = failure;

        public override DenseTensor<float> ToDenseTensor() => throw failure;
        public override Tensor<float> Clone()
        {
            CloneCalls++;
            return base.Clone();
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FatalConversionFailuresPropagateWithoutCloning(bool stackExhaustion)
    {
        // These are controlled exceptions, not actual resource exhaustion.
        Exception failure = stackExhaustion ? new StackOverflowException("controlled") : new OutOfMemoryException("controlled");
        var input = new ConversionFailure(failure);
        Assert.Same(failure, Record.Exception(() => CPUExecutionProvider.Constant(input, null)));
        Assert.Equal(0, input.CloneCalls);

        var node = new Node { Op = OpType.Constant, OpTypeName = "Constant", Domain = "", OpsetVersion = 18,
            Inputs = [], Outputs = ["y"], Attributes = new() { ["value"] = input } };
        Assert.Same(failure, Record.Exception(() => node.Execute(new ComputationalGraph(), ExecutionProvider.CPU, null)));
        Assert.Equal(0, input.CloneCalls);
    }

    [Fact]
    public void NonfatalConversionFailureRetainsOwnedCloneFallback()
    {
        var input = new ConversionFailure(new NotSupportedException("controlled"));
        var first = CPUExecutionProvider.Constant(input, null);
        var second = CPUExecutionProvider.Constant(input, null);
        Assert.Equal(OpStatus.Success, first.Status);
        Assert.Equal(OpStatus.Success, second.Status);
        var a = Assert.IsAssignableFrom<Tensor<float>>(first.Outputs[0]);
        var b = Assert.IsAssignableFrom<Tensor<float>>(second.Outputs[0]);
        Assert.Equal(2, input.CloneCalls);
        Assert.Equal(input.ToArray().Select(BitConverter.SingleToUInt32Bits), a.ToArray().Select(BitConverter.SingleToUInt32Bits));
        Assert.False(TensorAlias.SharesBackingMemory(input, a));
        Assert.False(TensorAlias.SharesBackingMemory(a, b));
        a.SetValue(1, 99f);
        Assert.Equal(2.5f, input.GetValue(1));
        Assert.Equal(2.5f, b.GetValue(1));
    }
}
