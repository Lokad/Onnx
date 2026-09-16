using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Direct differential for MathOps.Im2colRange stride-two gather against a
/// scalar reference replicating the legacy loop: every patch slot must match
/// bitwise (pure copies, no reassociation) across vector/scalar widths,
/// pad parities, null-line rows, and column-block tilings.
/// </summary>
public class Im2colRangeStride2Tests
{
    static float Pattern(int i) => ((i * 131 + 7) % 89 - 44) * 0.05f;

    static unsafe void ScalarRef(float* src, int srcC, int srcH, int srcW, int kH, int kW, int sH, int sW, int padY, int padX, int dstW, int colStart, int colCount, float* buf)
    {
        int dyFirst = colStart / dstW;
        int dyLast = (colStart + colCount - 1) / dstW;
        for (int sc = 0; sc < srcC; ++sc)
            for (int ky = 0; ky < kH; ++ky)
            {
                int row0 = ky - padY;
                for (int dy = dyFirst; dy <= dyLast; ++dy)
                {
                    int sy = row0 + dy * sH;
                    float* line = (uint)sy < (uint)srcH ? src + (sc * srcH + sy) * srcW : null;
                    int dxLo = dy == dyFirst ? colStart - dy * dstW : 0;
                    int dxHi = dy == dyLast ? colStart + colCount - dy * dstW : dstW;
                    for (int kx = 0; kx < kW; ++kx)
                    {
                        int col0 = kx - padX;
                        float* row = buf + ((sc * kH + ky) * kW + kx) * colCount - colStart;
                        for (int dx = dxLo; dx < dxHi; ++dx)
                        {
                            int sx = col0 + dx * sW;
                            row[dy * dstW + dx] = (line != null && (uint)sx < (uint)srcW) ? line[sx] : 0;
                        }
                    }
                }
            }
    }

    static unsafe void Check(int srcC, int srcH, int srcW, int padY, int padX, int colStart, int colCount)
    {
        int kH = 3, kW = 3, sH = 2, sW = 2;
        int dstW = (srcW + padX + padX - kW) / sW + 1;
        int dstH = (srcH + padY + padY - kH) / sH + 1;
        Assert.True(colStart + colCount <= dstW * dstH);
        var src = new float[srcC * srcH * srcW];
        for (int i = 0; i < src.Length; i++) src[i] = Pattern(i);
        var got = new float[srcC * kH * kW * colCount];
        var want = new float[srcC * kH * kW * colCount];
        fixed (float* ps = src)
        fixed (float* pg = got)
        fixed (float* pw = want)
        {
            Lokad.Onnx.MathOps.Im2colRange(ps, srcC, srcH, srcW, kH, kW, 1, 1, sH, sW, padY, padX, padY, padX, dstW, colStart, colCount, pg);
            ScalarRef(ps, srcC, srcH, srcW, kH, kW, sH, sW, padY, padX, dstW, colStart, colCount, pw);
        }
        for (int i = 0; i < got.Length; i++)
            Assert.True(got[i] == want[i], "buf[" + i + "]: got " + got[i] + " want " + want[i]);
    }

    [Theory]
    [InlineData(9, 1, 1)]
    [InlineData(15, 1, 1)]
    [InlineData(17, 1, 1)]
    [InlineData(33, 1, 1)]
    [InlineData(33, 0, 0)]
    [InlineData(33, 0, 2)]
    [InlineData(50, 1, 1)]
    [InlineData(65, 1, 1)]
    public void Stride2_FullTile_MatchesScalar(int srcW, int padY, int padX)
    {
        int dstW = (srcW + padX + padX - 3) / 2 + 1;
        int dstH = (20 + padY + padY - 3) / 2 + 1;
        Check(2, 20, srcW, padY, padX, 0, dstW * dstH);
    }

    [Theory]
    [InlineData(0, 102)]
    [InlineData(102, 102)]
    [InlineData(204, 46)]
    [InlineData(7, 31)]
    public void Stride2_SplitTiles_MatchesScalar(int colStart, int colCount)
    {
        // Transition-like tiling: 10x25 outputs in ~102-column blocks plus an
        // odd partial block crossing row boundaries mid-tile.
        Check(2, 20, 50, 1, 1, colStart, colCount);
    }

    [Fact]
    public void Stride2_ShortRows_MatchesScalar()
    {
        // Every source row is a null line or too short for vector loads.
        Check(2, 5, 9, 1, 1, 0, 15);
    }
    [SkippableFact]
    public void Deinterleave256Recipe_MatchesExpected()
    {
        // Pins the AVX2 index constants used by the Im2colRange stride-two
        // path. This recipe never executes on AVX512 hardware (the 512 lane
        // wins dispatch), so without this pin no test would cover the lane
        // that AVX2-only machines actually run.
        Skip.If(!Avx2.IsSupported, "AVX2 deinterleave recipe needs Avx2.");
        var v0 = Vector256.Create(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f);
        var v1 = Vector256.Create(9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f);
        var t = Avx.Shuffle(v0, v1, (byte)0x88);
        var idx = Vector256.Create(0, 1, 4, 5, 2, 3, 6, 7);
        var e = Avx2.PermuteVar8x32(t, idx);
        for (int i = 0; i < 8; i++)
            Assert.True(e.GetElement(i) == 2 * i + 1, "lane " + i + ": got " + e.GetElement(i));
    }

    [SkippableFact]
    public void Deinterleave512Recipe_MatchesExpected()
    {
        // Same pin for the AVX512 lane actually dispatched on this host.
        Skip.If(!Avx512F.IsSupported, "AVX512 deinterleave recipe needs Avx512F.");
        var a = Vector512.Create(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f);
        var b = Vector512.Create(17f, 18f, 19f, 20f, 21f, 22f, 23f, 24f, 25f, 26f, 27f, 28f, 29f, 30f, 31f, 32f);
        var t = Avx512F.Shuffle(a, b, (byte)0x88);
        var idx = Vector512.Create(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
        var e = Avx512F.PermuteVar16x32(t, idx);
        for (int i = 0; i < 16; i++)
            Assert.True(e.GetElement(i) == 2 * i + 1, "lane " + i + ": got " + e.GetElement(i));
    }
}