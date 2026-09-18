using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Lokad.Onnx;

public abstract partial class Tensor<T>
{
    // One vector per iteration keeps the existing cyclic-bias and tail rules.
    // Inlining erf removes vector arguments/results spilled across method calls.
    // This numeric loop should start optimized, including in short-lived requests.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe void BiasGeluSpanFloatInline(ReadOnlySpan<float> xs, ReadOnlySpan<float> bias, Span<float> ys)
    {
        int w = Vector<float>.Count;
        var half = new Vector<float>(0.5f);
        var one = Vector<float>.One;
        var scale = new Vector<float>(0.7071067811865476f);
        int M = bias.Length;
        if (M <= 1 || M % w != 0 || xs.Length != ys.Length)
        {
            int soff = 0;
            for (int i = 0; i < xs.Length; i++)
            {
                ys[i] = ScalarGelu(xs[i] + bias[soff]);
                soff++;
                if (soff >= M) soff = 0;
            }
            return;
        }
        fixed (float* px = xs, py = ys, pb = bias)
        {
            var xvec = (Vector<float>*)px;
            var yvec = (Vector<float>*)py;
            var bvec = (Vector<float>*)pb;
            int nvec = xs.Length / w;
            int bvecs = M / w;
            int boff = 0;
            for (int i = 0; i < nvec; i++)
            {
                var tv = xvec[i] + bvec[boff];
                yvec[i] = half * tv * (one + MathOps.ErfVectorInline(scale * tv));
                if (++boff >= bvecs) boff = 0;
            }
            int tail = nvec * w;
            int toff = tail % M;
            for (int i = tail; i < xs.Length; i++)
            {
                ys[i] = ScalarGelu(xs[i] + bias[toff]);
                toff++;
                if (toff >= M) toff = 0;
            }
        }
    }
}
