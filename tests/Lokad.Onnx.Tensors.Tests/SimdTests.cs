namespace Lokad.Onnx.Tensors.Tests;

using System.Buffers;
using Xunit;


using static Lokad.Onnx.MathOps;
public class SimdTests
{
    public SimdTests()
    {
        t_384_384_a = Tensor<float>.Rand(384, 384);
        t_384_384_b = Tensor<float>.Rand(384, 384);
        t_384_384_cr = Tensor<float>.MatMul2D_managed(t_384_384_a, t_384_384_b); 
        t_384_384_c.Fill(0.0f);
        t_384_384_c2.Fill(0.0f);
        ah = t_384_384_a.ToDenseTensor().Buffer.Pin();
        bh = t_384_384_b.ToDenseTensor().Buffer.Pin();
        ch = t_384_384_c.ToDenseTensor().Buffer.Pin();
        c2h = t_384_384_c2.ToDenseTensor().Buffer.Pin();
    }

    [Fact]
    public void CanMatMulManaged()
    {
        mm_managed(384, 384, 384, t_384_384_a.ToDenseTensor().Buffer, t_384_384_b.ToDenseTensor().Buffer, t_384_384_c.ToDenseTensor().Buffer);
        Assert.Equal(t_384_384_c, t_384_384_cr);
    }

    [Fact]
    public unsafe void CanMatMulVectorized()
    {
        mm_unsafe_vectorized(384, 384, 384, (float*) ah.Pointer, (float*)bh.Pointer, (float*)c2h.Pointer);
        Assert.Equal(t_384_384_c2, t_384_384_cr);
    }

    [Fact]
    public unsafe void CanMatMulVectorizedIntrinsics()
    {
        mm_unsafe_vectorized_intrinsics(384, 384, 384, (float*)ah.Pointer, (float*)bh.Pointer, (float*)c2h.Pointer);
        Assert.Equal(t_384_384_c2[0,1], t_384_384_cr[0,1], .00001f);
    }
    #region Fields
    Tensor<float> t_384_384_a = Tensor<float>.Zeros(0);
    Tensor<float> t_384_384_b = Tensor<float>.Zeros(0);
    Tensor<float> t_384_384_c = Tensor<float>.Zeros(384, 384);
    Tensor<float> t_384_384_c2 = Tensor<float>.Zeros(384, 384);
    Tensor<float> t_384_384_cr = Tensor<float>.Zeros(384, 384);
    MemoryHandle ah = new MemoryHandle();
    MemoryHandle bh = new MemoryHandle();
    MemoryHandle ch = new MemoryHandle();
    MemoryHandle c2h = new MemoryHandle();
    #endregion
}

