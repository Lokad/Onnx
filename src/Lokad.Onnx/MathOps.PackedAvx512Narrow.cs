using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
namespace Lokad.Onnx;

// Prepared full-panel remainder experiment. Keeps the existing packed layout,
// 12/8-row bulk tiles, row partition, FMA order and accumulating destinations.
// Opt in with PACKED_AVX512_ROWS and PACKED_AVX512_NARROW before process startup.
public partial class MathOps
{
    internal static unsafe bool TryPackedAvx512NarrowRows(int m, int n, int k, float* x, float* packed, float* dest)
    {
        if (!Avx512F.IsSupported || !Fma.IsSupported || m < 8 || n <= 0 || k <= 0 || k % 32 != 0)
            return false;
        int main = m / 12 * 12;
        int rest = m - main;
        if (rest == 1) { main -= 12; rest = 13; }
        if (rest == 4 && main >= 12) { main -= 12; rest = 16; }
        else if (rest == 2 && main >= 12) { main -= 12; rest = 14; }
        // Do not reroute shapes whose original composition already uses only wide tiles.
        int narrow = rest;
        while (narrow >= 8 && narrow != 9) narrow -= 8;
        if (narrow == 0) return false;
        if (main > 0)
            for (int column = 0; column < k; column += 32)
                PackedTile12(main, n, x, packed + column * n, dest, k, column);
        x += main * n;
        dest += main * k;
        int eights = 0;
        while (rest >= 8 && rest != 9) { eights += 8; rest -= 8; }
        if (eights > 0)
            for (int column = 0; column < k; column += 32)
                PackedTile8(eights, n, x, packed + column * n, dest, k, column);
        x += eights * n;
        dest += eights * k;
        if (rest == 0) return true;
        if (rest % 3 == 0)
            PackedNarrow3(rest, n, k, x, packed, dest);
        else if (rest % 2 == 0)
            PackedNarrow2(rest, n, k, x, packed, dest);
        else
        {
            PackedNarrow3(3, n, k, x, packed, dest);
            PackedNarrow2(rest - 3, n, k, x + 3 * n, packed, dest + 3 * k);
        }
        return true;
    }

    internal static unsafe void PackedNarrow3(int m, int n, int k, float* a, float* packed, float* c)
    {
        for (int column = 0; column < k; column += 32) PackedNarrowTile3(m, n, a, packed + column * n, c, k, column);
    }
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe void PackedNarrowTile3(int m, int n, float* a, float* panel, float* c, int k, int column)
    {
        for (int row = 0; row < m; row += 3)
        {
            float* group = c + row * k + column;
            float* aGroup = a + row * n;
            float* a0 = aGroup + 0 * n;
            float* a1 = aGroup + 1 * n;
            float* a2 = aGroup + 2 * n;
            Vector512<float> c00 = Avx512F.LoadVector512(group + 0 * k + 0);
            Vector512<float> c01 = Avx512F.LoadVector512(group + 0 * k + 16);
            Vector512<float> c10 = Avx512F.LoadVector512(group + 1 * k + 0);
            Vector512<float> c11 = Avx512F.LoadVector512(group + 1 * k + 16);
            Vector512<float> c20 = Avx512F.LoadVector512(group + 2 * k + 0);
            Vector512<float> c21 = Avx512F.LoadVector512(group + 2 * k + 16);
            for (int j = 0; j < n; j++)
            {
                var p = (Vector512<float>*)(panel + j * 32);
                var b0 = p[0]; var b1 = p[1];
                var aa = Vector512.Create(a0[j]);
                c00 = Avx512F.FusedMultiplyAdd(b0, aa, c00);
                c01 = Avx512F.FusedMultiplyAdd(b1, aa, c01);
                aa = Vector512.Create(a1[j]);
                c10 = Avx512F.FusedMultiplyAdd(b0, aa, c10);
                c11 = Avx512F.FusedMultiplyAdd(b1, aa, c11);
                aa = Vector512.Create(a2[j]);
                c20 = Avx512F.FusedMultiplyAdd(b0, aa, c20);
                c21 = Avx512F.FusedMultiplyAdd(b1, aa, c21);
            }
            Avx512F.Store(group + 0 * k + 0, c00);
            Avx512F.Store(group + 0 * k + 16, c01);
            Avx512F.Store(group + 1 * k + 0, c10);
            Avx512F.Store(group + 1 * k + 16, c11);
            Avx512F.Store(group + 2 * k + 0, c20);
            Avx512F.Store(group + 2 * k + 16, c21);
        }
    }
    internal static unsafe void PackedNarrow2(int m, int n, int k, float* a, float* packed, float* c)
    {
        for (int column = 0; column < k; column += 32) PackedNarrowTile2(m, n, a, packed + column * n, c, k, column);
    }
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe void PackedNarrowTile2(int m, int n, float* a, float* panel, float* c, int k, int column)
    {
        for (int row = 0; row < m; row += 2)
        {
            float* group = c + row * k + column;
            float* aGroup = a + row * n;
            float* a0 = aGroup + 0 * n;
            float* a1 = aGroup + 1 * n;
            Vector512<float> c00 = Avx512F.LoadVector512(group + 0 * k + 0);
            Vector512<float> c01 = Avx512F.LoadVector512(group + 0 * k + 16);
            Vector512<float> c10 = Avx512F.LoadVector512(group + 1 * k + 0);
            Vector512<float> c11 = Avx512F.LoadVector512(group + 1 * k + 16);
            for (int j = 0; j < n; j++)
            {
                var p = (Vector512<float>*)(panel + j * 32);
                var b0 = p[0]; var b1 = p[1];
                var aa = Vector512.Create(a0[j]);
                c00 = Avx512F.FusedMultiplyAdd(b0, aa, c00);
                c01 = Avx512F.FusedMultiplyAdd(b1, aa, c01);
                aa = Vector512.Create(a1[j]);
                c10 = Avx512F.FusedMultiplyAdd(b0, aa, c10);
                c11 = Avx512F.FusedMultiplyAdd(b1, aa, c11);
            }
            Avx512F.Store(group + 0 * k + 0, c00);
            Avx512F.Store(group + 0 * k + 16, c01);
            Avx512F.Store(group + 1 * k + 0, c10);
            Avx512F.Store(group + 1 * k + 16, c11);
        }
    }
}
