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
}