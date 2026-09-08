namespace Lokad.Onnx.Bench;

using System;
using System.Numerics;

// Naive 2D matrix-multiplication reference kernels, moved verbatim from the
// shipped core (MathOps) so benchmarks keep their comparison baselines
// without growing the distributed assembly. Production dispatch uses the
// unsafe scalar/SIMD/intrinsics kernels in the core, never these.

static class ReferenceKernels
{
    /// <summary>
    /// Matrix multiplication.
    /// </summary>
    /// <param name="M">A rows.</param>
    /// <param name="N">A columns.</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="B">Right matrix.</param>
    /// <param name="C">Result matrix.</param>
    public static void mm_managed(int M,
                          int N,
                          int K,
                          Memory<float> mA,
                          Memory<float> mB,
                          Memory<float> mC)
    {
        var A = mA.Span;
        var B = mB.Span;
        var C = mC.Span;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; ++j)
            {
                var a = A[i * N + j];
                var bp = j * K;
                var cp = i * K;

                for (int k = 0; k < K; ++k)
                {
                    C[cp + k] += a * B[bp + k];
                }
            }
        }
    }

    public static void mm_managed(int M,
                  int N,
                  int K,
                  Memory<int> mA,
                  Memory<int> mB,
                  Memory<int> mC)
    {
        var A = mA.Span;
        var B = mB.Span;
        var C = mC.Span;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; ++j)
            {
                var a = A[i * N + j];
                var bp = j * K;
                var cp = i * K;
                
                for (int k = 0; k < K; ++k)
                {
                    C[cp + k] +=  a * B[bp + k];
                }
            }
        }
    }

    public static void mm_vectorized(int M,
                 int N,
                 int K,
                 Memory<int> mA,
                 Memory<int> mB,
                 Memory<int> mC)
    {
        var v = Vector<int>.Count;
        var A = mA.Span;
        var B = mB.Span;
        var C = mC.Span;

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; ++j)
            {
                var a = A[i * N + j];
                var bp = j * K;
                var cp = i * K;

                for (int k = 0; k <= K - v; k+=v)
                {
                    var Bk = B.Slice(bp + k, v);
                    var Ck = C.Slice(cp + k, v);
                    Vector<int> vec1 = new Vector<int>(a);
                    Vector<int> vec2 = new Vector<int>(Bk);
                    Vector<int> vec3 = Vector.Multiply(vec1, vec2);
                    Vector<int> vec4 = new Vector<int>(Ck);
                    Vector<int> vec5 = Vector.Add(vec3, vec4);
                    vec5.CopyTo(Ck);
                }
            }
        }
    }

    public static void mm_vectorized(int M,
                 int N,
                 int K,
                 Memory<float> mA,
                 Memory<float> mB,
                 Memory<float> mC)
    {
        var v = Vector<float>.Count;
        var A = mA.Span;
        var B = mB.Span;
        var C = mC.Span;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; ++j)
            {
                var a = A[i * N + j];
                var bp = j * K;
                var cp = i * K;

                for (int k = 0; k <= K - v; k += v)
                {
                    var Bk = B.Slice(bp + k, v);
                    var Ck = C.Slice(cp + k, v);
                    
                    Vector<float> vec1 = new Vector<float>(a);
                    Vector<float> vec2 = new Vector<float>(Bk);
                    Vector<float> vec3 = Vector.Multiply(vec1, vec2);
                    Vector<float> vec4 = new Vector<float>(Ck);
                    Vector<float> vec5 = Vector.Add(vec3, vec4);
                    vec5.CopyTo(Ck);
                }
            }
        }
    }
}
