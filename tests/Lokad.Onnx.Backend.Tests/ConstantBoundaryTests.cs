using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// First direct coverage of the Constant op entry: scalar/list payloads,
/// tensor passthrough, and the honest refusals, against ORT 1.29 probes.
/// </summary>
public class ConstantBoundaryTests
{
    [Fact]
    public void ScalarAndListPayloads_MatchOrt()
    {
        // ORT 1.29 materializes every numeric scalar/list payload
        // (probed: value_float scalar, value_ints list); each form
        // round-trips exactly here.
        var rf = CPU.Constant(2.5f, null);
        Assert.Equal(OpStatus.Success, rf.Status);
        Assert.Equal(new float[] { 2.5f }, ((Tensor<float>)rf.Outputs[0]).ToArray());
        var rfa = CPU.Constant(new float[] { 1f, 2f, 3f }, null);
        Assert.Equal(OpStatus.Success, rfa.Status);
        Assert.Equal(new float[] { 1f, 2f, 3f }, ((Tensor<float>)rfa.Outputs[0]).ToArray());
        var ri = CPU.Constant(7, null);
        Assert.Equal(OpStatus.Success, ri.Status);
        Assert.Equal(new int[] { 7 }, ((Tensor<int>)ri.Outputs[0]).ToArray());
        var ria = CPU.Constant(new int[] { 1, -2 }, null);
        Assert.Equal(OpStatus.Success, ria.Status);
        Assert.Equal(new int[] { 1, -2 }, ((Tensor<int>)ria.Outputs[0]).ToArray());
        var rl = CPU.Constant(9L, null);
        Assert.Equal(OpStatus.Success, rl.Status);
        Assert.Equal(new long[] { 9L }, ((Tensor<long>)rl.Outputs[0]).ToArray());
        var rla = CPU.Constant(new long[] { 8L, -8L }, null);
        Assert.Equal(OpStatus.Success, rla.Status);
        Assert.Equal(new long[] { 8L, -8L }, ((Tensor<long>)rla.Outputs[0]).ToArray());
    }

    [Fact]
    public void TensorPayload_ClonesThrough()
    {
        // Tensor-valued constants of any dtype (e.g. double) ride the
        // clone path; ORT 1.29 returns the same values.
        var src = DenseTensor<double>.OfValues(new double[] { 1.5, -2.5 });
        var r = CPU.Constant(src, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<double>)r.Outputs[0];
        Assert.Equal(new double[] { 1.5, -2.5 }, y.ToArray());
        Assert.False(ReferenceEquals(src, y));
    }

    [Fact]
    public void MissingValue_FailsCleanly()
    {
        // ORT 1.29 refuses a valueless Constant at load; the provider
        // reports the missing attribute instead of throwing.
        var r = CPU.Constant(null, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("value", r.Message ?? "");
    }

    [Fact]
    public void UnsupportedPayloads_FailHonestly()
    {
        // No string tensor type exists in the engine, and boxed-double
        // scalar attributes have no producer (tensor doubles ride the
        // ITensor arm above); both fail with NotSupported while ORT
        // 1.29 materializes them (probed: string, valueless excluded).
        var rd = CPU.Constant(2.5, null);
        Assert.Equal(OpStatus.Failure, rd.Status);
        var rs = CPU.Constant("abc", null);
        Assert.Equal(OpStatus.Failure, rs.Status);
    }
}