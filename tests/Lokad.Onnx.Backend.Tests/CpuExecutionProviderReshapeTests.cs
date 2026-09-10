using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests
{
    public class CpuExecutionProviderReshapeTests
    {
        [Fact]
        public void CanReshape()
        {
            var X = DenseTensor<int>.Ones(2, 3, 4);
            var s = DenseTensor<long>.OfValues(new long[] { 4, 2, 3 });
            var r = CPU.Reshape(X, s, null, null);
            Assert.Equal(OpStatus.Success, r.Status);
            Assert.Equal(r.Outputs![0].Dims, new int[3] { 4, 2, 3 });

            s = DenseTensor<long>.OfValues(new long[] { -1, 2, 3, 4 });
            r = CPU.Reshape(X, s, null, null);
            Assert.Equal(OpStatus.Success, r.Status);
            Assert.Equal(r.Outputs![0].Dims, new int[4] { 1, 2, 3, 4 });

            r = CPU.Reshape((ITensor) X, null, null, null);
            Assert.Equal(OpStatus.Failure, r.Status);
           
            Assert.Throws<ArgumentException>(() => CPU.Reshape((ITensor)X, DenseTensor<long>.OfValues(new long[,] { { 2, 2 }, { 2, 1 } }), null, null));
        }
    [Fact]
    public void ReshapeZeroVolume_Inference()
    {
        // ORT 1.29: zero infers zero, never divides.
        Assert.Equal(new int[] { 0 }, DimsOf(DenseTensor<float>.OfShape(0), new long[] { -1 }));
        Assert.Equal(new int[] { 2, 0 }, DimsOf(DenseTensor<float>.OfShape(2, 0), new long[] { 0, -1 }));
        Assert.Equal(new int[] { 0 }, DimsOf(DenseTensor<float>.OfShape(2, 0), new long[] { -1 }));
        Assert.Equal(new int[] { 0 }, DimsOf(DenseTensor<float>.OfShape(0), new long[] { 0 }));
    }

    [Fact]
    public void ReshapeDoubleInfer_Throws()
    {
        // ORT 1.29 rejects two -1 dims; Lokad throws descriptively.
        var x = DenseTensor<float>.OfValues(new float[] { 1f, 2f, 3f, 4f });
        Assert.Throws<System.ArgumentException>(() => CPU.Reshape(x, DenseTensor<long>.OfValues(new long[] { -1, -1 }), false, null));
    }

    static int[] DimsOf(Tensor<float> x, long[] shape)
    {
        var r = CPU.Reshape(x, DenseTensor<long>.OfValues(shape), false, null);
        Assert.Equal(OpStatus.Success, r.Status);
        return ((Tensor<float>)r.Outputs[0]).Dimensions.ToArray();
    }

    [Fact]
    public void SqueezeUnsqueeze_Sub32Data()
    {
        // The INumericTensor view path is dtype-generic; pin sub-32 data.
        var x = DenseTensor<sbyte>.OfValues(new sbyte[,] { { 1, -2 } });
        var sq = CPU.Squeeze(x, DenseTensor<long>.OfValues(new long[] { 0L }), null);
        Assert.Equal(OpStatus.Success, sq.Status);
        Assert.Equal(new sbyte[] { 1, -2 }, ((Tensor<sbyte>)sq.Outputs[0]).ToArray());
        var un = CPU.Unsqueeze(DenseTensor<byte>.OfValues(new byte[] { 7, 8 }), DenseTensor<long>.OfValues(new long[] { 0L }), null);
        Assert.Equal(OpStatus.Success, un.Status);
        var uy = (Tensor<byte>)un.Outputs[0];
        Assert.Equal(new int[] { 1, 2 }, uy.Dimensions.ToArray());
        Assert.Equal(new byte[] { 7, 8 }, uy.ToArray());
    }
    }
}
