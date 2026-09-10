using System;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Float16/bfloat16 movement arms complete the half-dtype rows the arithmetic
/// refusal pins leave open. Float16 expectations carry ORT 1.29 values;
/// bfloat16 movement is verbatim (ORT kernels verified through Cast tails),
/// while bfloat16 Tile/Expand refusal matches ORT NOT_IMPLEMENTED.
/// </summary>
public class HalfMovementTests
{
    static DenseTensor<Half> HBase() =>
        DenseTensor<Half>.OfValues(new Half[] { (Half)1f, (Half)2f, (Half)3f, (Half)4f }, new int[] { 2, 2 });

    static DenseTensor<BFloat16> BBase() =>
        DenseTensor<BFloat16>.OfValues(new BFloat16[] { (BFloat16)1f, (BFloat16)2f, (BFloat16)3f, (BFloat16)4f }, new int[] { 2, 2 });

    [Fact]
    public void HalfReshape_MatchesOrt()
    {
        // ORT 1.29 float16: [[1, 2], [3, 4]] to [4].
        var r = CPU.Reshape(HBase(), DenseTensor<long>.OfValues(new long[] { 4 }), false, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new Half[] { (Half)1f, (Half)2f, (Half)3f, (Half)4f }, ((Tensor<Half>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void HalfTranspose_MatchesOrt()
    {
        // ORT 1.29 float16: perm [1, 0] over [[1, 2], [3, 4]].
        var r = CPU.Transpose(HBase(), new int[] { 1, 0 }, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new Half[] { (Half)1f, (Half)3f, (Half)2f, (Half)4f }, ((Tensor<Half>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void HalfConcat_MatchesOrt()
    {
        // ORT 1.29 float16: [[1, 2], [3, 4]] plus [[5, 6]] on axis 0.
        var y = DenseTensor<Half>.OfValues(new Half[] { (Half)5f, (Half)6f }, new int[] { 1, 2 });
        var r = CPU.Concat(new ITensor[] { HBase(), y }, 0, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new Half[] { (Half)1f, (Half)2f, (Half)3f, (Half)4f, (Half)5f, (Half)6f }, ((Tensor<Half>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void HalfSlice_MatchesOrt()
    {
        // ORT 1.29 runs float16 Slice (probed full-tensor); a row window is
        // verbatim movement of the same kernel path.
        var r = CPU.Slice(HBase(), DenseTensor<long>.OfValues(new long[] { 0 }), DenseTensor<long>.OfValues(new long[] { 1 }), DenseTensor<long>.OfValues(new long[] { 0 }), null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new Half[] { (Half)1f, (Half)2f }, ((Tensor<Half>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void HalfGather_MatchesOrt()
    {
        // ORT 1.29 float16: rows [1, 0] of [[1, 2], [3, 4]].
        var r = CPU.Gather(HBase(), DenseTensor<int>.OfValues(new int[] { 1, 0 }), 0, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new Half[] { (Half)3f, (Half)4f, (Half)1f, (Half)2f }, ((Tensor<Half>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void BFloat16Reshape_MatchesValues()
    {
        // ORT bfloat16 Reshape kernel verified through a Cast tail; values
        // move verbatim.
        var r = CPU.Reshape(BBase(), DenseTensor<long>.OfValues(new long[] { 4 }), false, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new BFloat16[] { (BFloat16)1f, (BFloat16)2f, (BFloat16)3f, (BFloat16)4f }, ((Tensor<BFloat16>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void BFloat16Transpose_MatchesValues()
    {
        // ORT bfloat16 Transpose kernel verified through a Cast tail.
        var r = CPU.Transpose(BBase(), new int[] { 1, 0 }, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new BFloat16[] { (BFloat16)1f, (BFloat16)3f, (BFloat16)2f, (BFloat16)4f }, ((Tensor<BFloat16>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void BFloat16Concat_MatchesValues()
    {
        // ORT bfloat16 Concat kernel verified through a Cast tail.
        var y = DenseTensor<BFloat16>.OfValues(new BFloat16[] { (BFloat16)5f, (BFloat16)6f }, new int[] { 1, 2 });
        var r = CPU.Concat(new ITensor[] { BBase(), y }, 0, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new BFloat16[] { (BFloat16)1f, (BFloat16)2f, (BFloat16)3f, (BFloat16)4f, (BFloat16)5f, (BFloat16)6f }, ((Tensor<BFloat16>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void BFloat16Slice_MatchesValues()
    {
        // ORT bfloat16 Slice kernel verified through a Cast tail.
        var r = CPU.Slice(BBase(), DenseTensor<long>.OfValues(new long[] { 0 }), DenseTensor<long>.OfValues(new long[] { 1 }), DenseTensor<long>.OfValues(new long[] { 0 }), null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new BFloat16[] { (BFloat16)1f, (BFloat16)2f }, ((Tensor<BFloat16>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void BFloat16Gather_MatchesValues()
    {
        // ORT bfloat16 Gather kernel verified through a Cast tail.
        var r = CPU.Gather(BBase(), DenseTensor<int>.OfValues(new int[] { 1, 0 }), 0, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new BFloat16[] { (BFloat16)3f, (BFloat16)4f, (BFloat16)1f, (BFloat16)2f }, ((Tensor<BFloat16>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void BFloat16Split_MatchesValues()
    {
        // ORT bfloat16 Split kernel verified through a Cast tail; the
        // provider arm mirrors the float16 densify path exactly.
        var r = CPU.Split(DenseTensor<BFloat16>.OfValues(new BFloat16[] { (BFloat16)1f, (BFloat16)2f, (BFloat16)3f, (BFloat16)4f }, new int[] { 1, 4 }), DenseTensor<long>.OfValues(new long[] { 2L, 2L }), 1, null, null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new BFloat16[] { (BFloat16)1f, (BFloat16)2f }, ((Tensor<BFloat16>)r.Outputs![0]).ToArray());
        Assert.Equal(new BFloat16[] { (BFloat16)3f, (BFloat16)4f }, ((Tensor<BFloat16>)r.Outputs![1]).ToArray());
    }

    [Fact]
    public void BFloat16TileExpand_RefusedCleanly()
    {
        // ORT 1.29 has no CPU kernel for bfloat16 Tile or Expand
        // (NOT_IMPLEMENTED, probed); the provider refusal matches.
        var x = DenseTensor<BFloat16>.OfValues(new BFloat16[] { (BFloat16)1f, (BFloat16)2f }, new int[] { 1, 2 });
        Assert.Equal(OpStatus.Failure, CPU.Tile(x, DenseTensor<long>.OfValues(new long[] { 1L, 2L }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Expand(x, DenseTensor<long>.OfValues(new long[] { 2L, 2L }), null).Status);
    }
}
