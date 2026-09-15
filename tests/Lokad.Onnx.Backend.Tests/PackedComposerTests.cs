using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Covers the tile-major packed row-group composer: extracted tile entries match
/// their full methods bit for bit, and the composed per-tile chain matches the
/// legacy grouped calls bit for bit on mixed 12/8 decompositions with tails.
/// </summary>
public class PackedComposerTests
{
    static float[] Range(float start, float step, int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = start + step * i;
        return v;
    }

    static void Pack(int n, int k, float[] b, float[] p)
    {
        unsafe
        {
            fixed (float* bp = b, pp = p)
            {
                MathOps.PackPanelsB(n, k, bp, pp);
            }
        }
    }

    static void Tile12(int n, int k, float[] a, int aOff, float[] p, int pOff, float[] c, int cOff, int kb)
    {
        unsafe
        {
            fixed (float* ap = a, pp = p, cp = c)
            {
                MathOps.mm_avx512_12x32packed_tile(12, n, ap + aOff, pp + pOff, cp + cOff, k, kb, false);
            }
        }
    }

    static void Tile8(int m, int n, int k, float[] a, int aOff, float[] p, int pOff, float[] c, int cOff, int kb)
    {
        unsafe
        {
            fixed (float* ap = a, pp = p, cp = c)
            {
                MathOps.mm_avx512_8x32packed_tile(m, n, ap + aOff, pp + pOff, cp + cOff, k, kb, false);
            }
        }
    }

    static void Full12(int m, int n, int k, float[] a, int aOff, float[] p, float[] c, int cOff)
    {
        unsafe
        {
            fixed (float* ap = a, pp = p, cp = c)
            {
                MathOps.mm_unsafe_vectorized_avx512_12x32packed(m, n, k, ap + aOff, pp, cp + cOff, false);
            }
        }
    }

    static void Full8(int m, int n, int k, float[] a, int aOff, float[] p, float[] c, int cOff)
    {
        unsafe
        {
            fixed (float* ap = a, pp = p, cp = c)
            {
                MathOps.mm_unsafe_vectorized_avx512_8x32packed(m, n, k, ap + aOff, pp, cp + cOff, false);
            }
        }
    }

    static void AssertBitsEqual(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(
                System.BitConverter.SingleToInt32Bits(expected[i]),
                System.BitConverter.SingleToInt32Bits(actual[i]));
        }
    }

    [Fact]
    public void TileEntries_MatchFullMethods()
    {
        const int n = 256;
        const int k = 512;
        var a = Range(-1f, 0.01f, 24 * n);
        var b = Range(-2f, 0.001f, n * k);
        var p = new float[n * k];
        Pack(n, k, b, p);
        var cFull12 = new float[24 * k];
        var cTile12 = new float[24 * k];
        Full12(24, n, k, a, 0, p, cFull12, 0);
        var cFull8 = new float[16 * k];
        var cTile8 = new float[16 * k];
        Full8(16, n, k, a, 0, p, cFull8, 0);
        int tileStep = 32;
        int tiles = k / tileStep;
        for (int tb = 0; tb < tiles; tb++)
        {
            int kb = tb * tileStep;
            int pOff = tb * n * tileStep;
            Tile12(n, k, a, 0, p, pOff, cTile12, 0, kb);
            Tile12(n, k, a, 12 * n, p, pOff, cTile12, 12 * k, kb);
            Tile8(16, n, k, a, 0, p, pOff, cTile8, 0, kb);
        }
        AssertBitsEqual(cFull12, cTile12);
        AssertBitsEqual(cFull8, cTile8);
    }

    [Fact]
    public void ComposedChain_MatchesGroupedCalls_JointGeometry()
    {
        const int m = 40;
        const int n = 640;
        const int k = 8198;
        var a = Range(-1f, 0.0001f, m * n);
        var b = Range(0.5f, -0.00001f, n * k);
        var p = new float[n * k];
        Pack(n, k, b, p);
        var legacy = new float[m * k];
        Full12(24, n, k, a, 0, p, legacy, 0);
        Full8(16, n, k, a, 24 * n, p, legacy, 24 * k);
        var composed = new float[m * k];
        int tileStep = 32;
        int blocked = k - (k % tileStep);
        int tiles = blocked / tileStep;
        for (int tb = 0; tb < tiles; tb++)
        {
            int kb = tb * tileStep;
            int pOff = tb * n * tileStep;
            Tile12(n, k, a, 0, p, pOff, composed, 0, kb);
            Tile12(n, k, a, 12 * n, p, pOff, composed, 12 * k, kb);
            Tile8(16, n, k, a, 24 * n, p, pOff, composed, 24 * k, kb);
        }
        int remCols = k - blocked;
        if (remCols > 0)
        {
            Tail12(n, k, a, 0, p, composed, 0, blocked, tiles, remCols);
            Tail12(n, k, a, 12 * n, p, composed, 12 * k, blocked, tiles, remCols);
            Tail8(16, n, k, a, 24 * n, p, composed, 24 * k, blocked, tiles, remCols);
        }
        AssertBitsEqual(legacy, composed);
    }


    static double[] Oracle(float[] a, float[] b, int m, int k, int n)
    {
        var c = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double acc = 0.0;
                for (int l = 0; l < k; l++) acc += (double)a[i * k + l] * b[l * n + j];
                c[i * n + j] = acc;
            }
        return c;
    }

    static void ComposerCase(int m, int k, int n)
    {
        var rnd = new System.Random(7711 + m);
        var x = Tensor<float>.Zeros(m, k).ToDenseTensor();
        var w = Tensor<float>.Zeros(k, n).ToDenseTensor();
        for (int i = 0; i < x.Length; i++) x.SetValue(i, rnd.NextSingle());
        for (int i = 0; i < w.Length; i++) w.SetValue(i, rnd.NextSingle());
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "mm-composer";
        graph.Inputs["x"] = x;
        graph.Initializers["w"] = w;
        graph.Outputs["y"] = Tensor<float>.Zeros(m, n).ToDenseTensor();
        graph.Nodes.Add(new Node
        {
            Name = "mm",
            Op = OpType.MatMul,
            OpTypeName = "MatMul",
            Domain = "",
            Inputs = new[] { "x", "w" },
            Outputs = new[] { "y" },
        });
        graph.RefreshLifetimeAnalysis();
        Assert.True(graph.Initializers.ContainsKey("packed:w"), "m=" + m + " did not pack.");
        var user = new Dictionary<string, ITensor> { ["x"] = x };
        Assert.True(graph.Execute(user, true), graph.LastErrorMessage + " m=" + m);
        var got = ((Tensor<float>)graph.Outputs["y"]).ToArray();
        var want = Oracle(x.ToArray(), w.ToArray(), m, k, n);
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++)
        {
            double tol = 1e-5 * (1.0 + System.Math.Abs(want[i]));
            Assert.True(System.Math.Abs(got[i] - want[i]) <= tol, "m=" + m + " index=" + i);
        }
    }

    [Fact]
    public void MixedDecompositionComposer_AgreesWithOracle()
    {
        ComposerCase(20, 256, 2048);
        ComposerCase(40, 256, 2048);
        ComposerCase(31, 1024, 1024);
        ComposerCase(5, 640, 2560);
    }

    [Fact]
    public void NarrowTiles_MatchFullMethods()
    {
        const int n = 256;
        const int k = 512;
        var a = Range(-1f, 0.01f, 8 * n);
        var b = Range(-2f, 0.001f, n * k);
        var p = new float[n * k];
        Pack(n, k, b, p);
        var cFull3 = new float[3 * k];
        var cTile3 = new float[3 * k];
        Full3(3, n, k, a, 0, p, cFull3, 0);
        var cFull2 = new float[4 * k];
        var cTile2 = new float[4 * k];
        Full2(4, n, k, a, 0, p, cFull2, 0);
        int tileStep = 32;
        int tiles = k / tileStep;
        for (int tb = 0; tb < tiles; tb++)
        {
            int kb = tb * tileStep;
            int pOff = tb * n * tileStep;
            Tile3(n, k, a, 0, p, pOff, cTile3, 0, kb);
            Tile2(4, n, k, a, 0, p, pOff, cTile2, 0, kb);
        }
        AssertBitsEqual(cFull3, cTile3);
        AssertBitsEqual(cFull2, cTile2);
    }

    [Fact]
    public void NarrowComposedChain_MatchesGroupedCalls()
    {
        const int m = 31;
        const int n = 1024;
        const int k = 1024;
        var a = Range(-1f, 0.0001f, m * n);
        var b = Range(0.5f, -0.00001f, n * k);
        var p = new float[n * k];
        Pack(n, k, b, p);
        var legacy = new float[m * k];
        Full12(24, n, k, a, 0, p, legacy, 0);
        Full3(3, n, k, a, 24 * n, p, legacy, 24 * k);
        Full2(4, n, k, a, 27 * n, p, legacy, 27 * k);
        var composed = new float[m * k];
        int tileStep = 32;
        int tiles = k / tileStep;
        for (int tb = 0; tb < tiles; tb++)
        {
            int kb = tb * tileStep;
            int pOff = tb * n * tileStep;
            Tile12(n, k, a, 0, p, pOff, composed, 0, kb);
            Tile12(n, k, a, 12 * n, p, pOff, composed, 12 * k, kb);
            Tile3(n, k, a, 24 * n, p, pOff, composed, 24 * k, kb);
            Tile2(4, n, k, a, 27 * n, p, pOff, composed, 27 * k, kb);
        }
        AssertBitsEqual(legacy, composed);
    }

    static void Full3(int m, int n, int k, float[] a, int aOff, float[] p, float[] c, int cOff)
    {
        unsafe
        {
            fixed (float* ap = a, pp = p, cp = c)
            {
                MathOps.mm_unsafe_vectorized_intrinsics_3x4packed(m, n, k, ap + aOff, pp, cp + cOff, false);
            }
        }
    }

    static void Full2(int m, int n, int k, float[] a, int aOff, float[] p, float[] c, int cOff)
    {
        unsafe
        {
            fixed (float* ap = a, pp = p, cp = c)
            {
                MathOps.mm_unsafe_vectorized_intrinsics_2x4packed(m, n, k, ap + aOff, pp, cp + cOff, false);
            }
        }
    }

    static void Tile3(int n, int k, float[] a, int aOff, float[] p, int pOff, float[] c, int cOff, int kb)
    {
        unsafe
        {
            fixed (float* ap = a, pp = p, cp = c)
            {
                MathOps.mm_3x4packed_tile(3, n, ap + aOff, pp + pOff, cp + cOff, k, kb, false);
            }
        }
    }

    static void Tile2(int m, int n, int k, float[] a, int aOff, float[] p, int pOff, float[] c, int cOff, int kb)
    {
        unsafe
        {
            fixed (float* ap = a, pp = p, cp = c)
            {
                MathOps.mm_2x4packed_tile(m, n, ap + aOff, pp + pOff, cp + cOff, k, kb, false);
            }
        }
    }


    static void Tail12(int n, int k, float[] a, int aOff, float[] p, float[] c, int cOff, int blocked, int tiles, int remCols)
    {
        unsafe
        {
            fixed (float* ap = a, pp = p, cp = c)
            {
                MathOps.mm_avx512_12x32packed_col_tail(12, n, k, ap + aOff, pp, cp + cOff, blocked, tiles, remCols, false);
            }
        }
    }

    static void Tail8(int m, int n, int k, float[] a, int aOff, float[] p, float[] c, int cOff, int blocked, int tiles, int remCols)
    {
        unsafe
        {
            fixed (float* ap = a, pp = p, cp = c)
            {
                MathOps.mm_avx512_8x32packed_col_tail(m, n, k, ap + aOff, pp, cp + cOff, blocked, tiles, remCols, false);
            }
        }
    }
}

