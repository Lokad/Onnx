#if NET11_0_OR_GREATER
namespace Lokad.Onnx;

using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

public abstract partial class Tensor<T> where T : unmanaged
{
    // Caller admits only dense, unpadded [N,C,H,1] kernel/stride-three pooling.
    static unsafe void MaxPoolTriples(Span<float> input, Span<float> output, int planes, int length, int outputs)
    {
        var first = Vector512.Create(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45);
        var second = first + Vector512.Create(1); var third = first + Vector512.Create(2);
        var firstHigh = first - Vector512.Create(16);
        var secondHigh = second - Vector512.Create(16); var thirdHigh = third - Vector512.Create(16);
        var firstMask = Vector512.GreaterThan(first, Vector512.Create(31)).AsSingle();
        var secondMask = Vector512.GreaterThan(second, Vector512.Create(31)).AsSingle();
        var thirdMask = Vector512.GreaterThan(third, Vector512.Create(31)).AsSingle();
        var initial = Vector512.Create(-float.MaxValue);
        fixed (float* source = input, destination = output)
        for (int plane = 0; plane < planes; plane++)
        {
            float* row = source + plane * length; float* target = destination + plane * outputs;
            int i = 0;
            for (; i + 16 <= outputs && i * 3 + 48 <= length; i += 16)
            {
                var a = Avx512F.LoadVector512(row + i * 3);
                var b = Avx512F.LoadVector512(row + i * 3 + 16);
                var c = Avx512F.LoadVector512(row + i * 3 + 32);
                var x = Vector512.ConditionalSelect(firstMask,
                    Avx512F.PermuteVar16x32x2(b, firstHigh, c), Avx512F.PermuteVar16x32x2(a, first, b));
                var y = Vector512.ConditionalSelect(secondMask,
                    Avx512F.PermuteVar16x32x2(b, secondHigh, c), Avx512F.PermuteVar16x32x2(a, second, b));
                var z = Vector512.ConditionalSelect(thirdMask,
                    Avx512F.PermuteVar16x32x2(b, thirdHigh, c), Avx512F.PermuteVar16x32x2(a, third, b));
                // x86 MAX selects its second operand on NaN or ties. Keeping
                // best second matches the original ordered strict comparison.
                var best = Avx512F.Max(x, initial);
                best = Avx512F.Max(y, best); best = Avx512F.Max(z, best);
                Avx512F.Store(target + i, best);
            }
            for (; i < outputs; i++)
            {
                float best = -float.MaxValue;
                for (int tap = 0; tap < 3 && i * 3 + tap < length; tap++)
                {
                    float value = row[i * 3 + tap];
                    if (value > best) best = value;
                }
                target[i] = best;
            }
        }
    }
}
#endif
