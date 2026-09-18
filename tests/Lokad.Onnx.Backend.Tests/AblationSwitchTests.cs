namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// The measurement-only ablation switches route to the legacy kernels without
/// changing default behavior. These tests set the fields directly (never the
/// environment) and always restore them, because the flags are process-wide;
/// other test classes must not execute while a switch is temporarily changed.
/// In particular, MaskedSoftmax's fused kernel stays current while its composite
/// reference calls the switchable public Softmax, so parallel mutation breaks
/// that test's intended bitwise comparison.
/// </summary>
[Collection("ProcessState")]
public class AblationSwitchTests
{
    [Fact]
    public void SoftmaxSwitchRoutesToLegacyKernel()
    {
        var rnd = new Random(1234);
        var x = new float[4 * 8];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rnd.NextDouble() * 8 - 4);
        bool saved = AblationSwitches.ForceLegacySoftmax;
        try
        {
            AblationSwitches.ForceLegacySoftmax = true;
            AblationSwitches.ResetLatches();
            var input = new DenseTensor<float>((float[])x.Clone(), new[] { 4, 8 });
            var routed = Tensor<float>.Softmax(input, 1, TensorExecutionOptions.Intrinsics, 13).ToDenseTensor().Buffer.Span.ToArray();
            Assert.True(AblationSwitches.LegacySoftmaxUsed, "legacy softmax latch was not set.");
            var direct = new float[x.Length];
            Tensor<float>.SoftmaxContiguousFloat((float[])x.Clone(), direct, 4, 8, true);
            Assert.Equal(direct, routed);
        }
        finally
        {
            AblationSwitches.ForceLegacySoftmax = saved;
            AblationSwitches.ResetLatches();
        }
    }

    [Fact]
    public void SoftmaxDefaultLeavesLegacyLatchClear()
    {
        bool saved = AblationSwitches.ForceLegacySoftmax;
        try
        {
            AblationSwitches.ForceLegacySoftmax = false;
            AblationSwitches.ResetLatches();
            var input = new DenseTensor<float>(new float[4 * 8], new[] { 4, 8 });
            Tensor<float>.Softmax(input, 1, TensorExecutionOptions.Intrinsics, 13);
            Assert.False(AblationSwitches.LegacySoftmaxUsed, "default path must not set the legacy latch.");
        }
        finally
        {
            AblationSwitches.ForceLegacySoftmax = saved;
            AblationSwitches.ResetLatches();
        }
    }

    [Fact]
    public void TransposeSwitchSkipsTiledFaceBitwiseIdentical()
    {
        var rnd = new Random(5678);
        var x = new float[2 * 3 * 5 * 7];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rnd.NextDouble() * 8 - 4);
        var unforced = Tensor<float>.Transpose(new DenseTensor<float>((float[])x.Clone(), new[] { 2, 3, 5, 7 }), new[] { 0, 1, 3, 2 }).ToDenseTensor().Buffer.Span.ToArray();
        bool saved = AblationSwitches.ForceLegacyTransposeFace;
        try
        {
            AblationSwitches.ForceLegacyTransposeFace = true;
            AblationSwitches.ResetLatches();
            var forced = Tensor<float>.Transpose(new DenseTensor<float>((float[])x.Clone(), new[] { 2, 3, 5, 7 }), new[] { 0, 1, 3, 2 }).ToDenseTensor().Buffer.Span.ToArray();
            Assert.True(AblationSwitches.LegacyTransposeFaceUsed, "transpose skip latch was not set.");
            Assert.Equal(unforced, forced);
        }
        finally
        {
            AblationSwitches.ForceLegacyTransposeFace = saved;
            AblationSwitches.ResetLatches();
        }
    }

    [Fact]
    public void TransposeDefaultLeavesSkipLatchClear()
    {
        bool saved = AblationSwitches.ForceLegacyTransposeFace;
        try
        {
            AblationSwitches.ForceLegacyTransposeFace = false;
            AblationSwitches.ResetLatches();
            Tensor<float>.Transpose(new DenseTensor<float>(new float[2 * 3 * 5 * 7], new[] { 2, 3, 5, 7 }), new[] { 0, 1, 3, 2 });
            Assert.False(AblationSwitches.LegacyTransposeFaceUsed, "default path must not set the skip latch.");
        }
        finally
        {
            AblationSwitches.ForceLegacyTransposeFace = saved;
            AblationSwitches.ResetLatches();
        }
    }
    [Fact]
    public void EnvMappingReadsExactOneValue()
    {
        const string name = "LOKAD_ONNX_ABLATION_TEST_PROBE";
        string? saved = System.Environment.GetEnvironmentVariable(name);
        try
        {
            System.Environment.SetEnvironmentVariable(name, "1");
            Assert.True(AblationSwitches.EnvIsSet(name));
            System.Environment.SetEnvironmentVariable(name, "0");
            Assert.False(AblationSwitches.EnvIsSet(name));
            System.Environment.SetEnvironmentVariable(name, null);
            Assert.False(AblationSwitches.EnvIsSet(name));
        }
        finally
        {
            System.Environment.SetEnvironmentVariable(name, saved);
        }
    }
}

