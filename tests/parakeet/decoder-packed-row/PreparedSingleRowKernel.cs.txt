namespace Lokad.Onnx;

using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>Single-row accumulation over the existing 32-column packed layout.</summary>
internal static unsafe class PreparedSingleRowKernel
{
    internal static void Multiply(int n, int k, float* a, float* packed, float* output)
    {
        const int Chunk = 32;
        int blocked = k - k % Chunk;
        for (int kb = 0; kb < blocked; kb += Chunk)
        {
            var cp = (Vector256<float>*)(output + kb);
            Vector256<float> c0 = cp[0];
            Vector256<float> c1 = cp[1];
            Vector256<float> c2 = cp[2];
            Vector256<float> c3 = cp[3];
            float* panel = packed + kb * n;
            for (int j = 0; j < n; ++j)
            {
                var av = Vector256.Create(a[j]);
                var bp = (Vector256<float>*)(panel + j * Chunk);
                c0 = Fma.MultiplyAdd(bp[0], av, c0);
                c1 = Fma.MultiplyAdd(bp[1], av, c1);
                c2 = Fma.MultiplyAdd(bp[2], av, c2);
                c3 = Fma.MultiplyAdd(bp[3], av, c3);
            }
            cp[0] = c0;
            cp[1] = c1;
            cp[2] = c2;
            cp[3] = c3;
        }
        int remaining = k - blocked;
        float* tail = packed + blocked * n;
        int ceiling = k / Vector256<float>.Count * Vector256<float>.Count;
        for (int column = blocked; column < ceiling; column += Vector256<float>.Count)
        {
            Vector256<float> c = *(Vector256<float>*)(output + column);
            for (int j = 0; j < n; ++j)
            {
                var bp = (Vector256<float>*)(tail + j * remaining + column - blocked);
                c = Fma.MultiplyAdd(bp[0], Vector256.Create(a[j]), c);
            }
            *(Vector256<float>*)(output + column) = c;
        }
        for (int column = ceiling; column < k; column++)
            for (int j = 0; j < n; ++j)
                output[column] += a[j] * tail[j * remaining + column - blocked];
    }
}
