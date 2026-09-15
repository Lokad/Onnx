using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

// Pins every remainder path of the packed row-group dispatch (12/8/6/3/2-row
// kernels plus the rem-1/2/4/7/9 adjustments) and the 32-column tail, through
// graph execution against a double oracle. The packed clone key is asserted
// present so the sweep provably exercises the packed path, not the fallback.
public class MatMulDispatchBoundaryTests
{
    const int Seed = 7711;

    static DenseTensor<float> FillRect(int rows, int cols, Random rnd)
    {
        var t = Tensor<float>.Zeros(rows, cols).ToDenseTensor();
        for (int i = 0; i < t.Length; i++) t.SetValue(i, rnd.NextSingle());
        return t;
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

    [Fact]
    public void RowGroupRemainders_AgreeWithOracle()
    {
        var rnd = new Random(Seed);
        var ms = new List<int>();
        for (int m = 1; m <= 40; m++) ms.Add(m);
        ms.Add(48);
        ms.Add(64);
        foreach (int m in ms)
        {
            foreach (int n in new[] { 8, 32, 40 })
            {
                const int k = 48;
                var x = FillRect(m, k, rnd);
                var w = FillRect(k, n, rnd);
                var graph = new ComputationalGraph();
                graph.Metadata["Name"] = "mm-dispatch";
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
                Assert.True(graph.Initializers.ContainsKey("packed:w"), "m=" + m + " n=" + n + " did not pack.");
                var user = new Dictionary<string, ITensor> { ["x"] = x };
                Assert.True(graph.Execute(user, true), graph.LastErrorMessage + " m=" + m + " n=" + n);
                var got = ((Tensor<float>)graph.Outputs["y"]).ToArray();
                var want = Oracle(x.ToArray(), w.ToArray(), m, k, n);
                Assert.Equal(want.Length, got.Length);
                for (int i = 0; i < want.Length; i++)
                {
                    double tol = 1e-5 * (1.0 + System.Math.Abs(want[i]));
                    Assert.True(System.Math.Abs(got[i] - want[i]) <= tol, "m=" + m + " n=" + n + " index=" + i);
                }
            }
        }
    }
}

