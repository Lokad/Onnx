namespace Lokad.Onnx.Tensors.Tests;

using System;
using System.Buffers;

using static Lokad.Onnx.MathOps;
public class TensorOpsSimdTests : IDisposable
{
    const int Seed = 20260904;

    static Tensor<float> SeededRand(int rows, int cols, int seed)
    {
        var t = Tensor<float>.Zeros(rows, cols);
        var rnd = new Random(seed);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                t[i, j] = rnd.NextSingle();
            }
        }
        return t;
    }

    public TensorOpsSimdTests()
    {
        t_384_384_a = SeededRand(384, 384, Seed);
        t_384_384_b = SeededRand(384, 384, Seed + 1);
        t_384_384_cr = ReferenceMatMul.Managed(t_384_384_a, t_384_384_b);
        t_384_384_c.Fill(0.0f);
        t_384_384_c2.Fill(0.0f);
        ah = t_384_384_a.ToDenseTensor().Buffer.Pin();
        bh = t_384_384_b.ToDenseTensor().Buffer.Pin();
        ch = t_384_384_c.ToDenseTensor().Buffer.Pin();
        c2h = t_384_384_c2.ToDenseTensor().Buffer.Pin();
    }

    public void Dispose()
    {
        ah.Dispose();
        bh.Dispose();
        ch.Dispose();
        c2h.Dispose();
    }

    [Fact]
    public void CanMatMulScalarDispatch()
    {
        var actual = Tensor<float>.MatMul2D(t_384_384_a, t_384_384_b, TensorExecutionOptions.Scalar);
        Assert.Equal(t_384_384_cr, actual);
    }

    [Fact]
    public unsafe void CanMatMulVectorized()
    {
        mm_unsafe_vectorized(384, 384, 384, (float*) ah.Pointer, (float*)bh.Pointer, (float*)c2h.Pointer);
        Assert.Equal(t_384_384_c2, t_384_384_cr);
    }

    [SkippableFact]
    public unsafe void CanMatMulVectorizedIntrinsics()
    {
        Skip.If(!System.Runtime.Intrinsics.X86.Fma.IsSupported, "x86 FMA not available on this machine.");
        mm_unsafe_vectorized_intrinsics(384, 384, 384, (float*)ah.Pointer, (float*)bh.Pointer, (float*)c2h.Pointer);
        Assert.Equal(t_384_384_c2[0,1], t_384_384_cr[0,1], .0002f);
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
