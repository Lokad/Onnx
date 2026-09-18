using System;
using System.Collections.Generic;
using System.Linq;
using Lokad.Onnx;

namespace Lokad.Onnx.Bench;

public sealed record GemmShape(string Name, int M, int N, int K, bool PreparedB, string Source);

public sealed record GemmMatrixResult(string Name, int M, int N, int K, bool PreparedB, string Route, double MaxScaled, bool Pass);

// W3 GEMM shape matrix: agreement plus dispatch routes, no timing.
// PreparedB shapes run a one-node MatMul graph with a W initializer (faithful
// prepared-pack dispatch); dynamic shapes call MatMul2D directly (attention-like).
// Reference is float64 MatMul; centered inputs keep the 1e-4 scaled gate sound
// even at K=8198. Route strings join multiple lane reports with + (sorted).
public static class GemmMatrix
{
    public const double Tolerance = 1e-4;

    static DenseTensor<float> FillF(int rows, int cols)
    {
        var t = DenseTensor<float>.OfShape(new int[] { rows, cols });
        var span = t.Buffer.Span;
        for (int i = 0; i < span.Length; i++) span[i] = (float)((i % 97) - 48) * 0.01f;
        return t;
    }

    static DenseTensor<double> FillD(int rows, int cols)
    {
        var t = DenseTensor<double>.OfShape(new int[] { rows, cols });
        var span = t.Buffer.Span;
        for (int i = 0; i < span.Length; i++) span[i] = (double)((i % 97) - 48) * 0.01;
        return t;
    }

    static string RouteOf(ComputationalGraph g)
    {
        if (g.LastProfile is null) return "none";
        foreach (var np in g.LastProfile)
        {
            int ri = np.Detail.IndexOf(" route=", StringComparison.Ordinal);
            if (ri >= 0) return np.Detail.Substring(ri + 7).Trim();
        }
        return "none";
    }

    public static List<GemmMatrixResult> Run(IEnumerable<GemmShape> shapes)
    {
        return Run(shapes, TensorExecutionOptions.Auto);
    }

    public static List<GemmMatrixResult> Run(IEnumerable<GemmShape> shapes, TensorExecutionOptions tensorOpts)
    {
        var results = new List<GemmMatrixResult>();
        foreach (var s in shapes) results.Add(RunOne(s, tensorOpts));
        return results;
    }

    static GemmMatrixResult RunOne(GemmShape s, TensorExecutionOptions tensorOpts)
    {
        Tensor<float> y;
        string route;
        if (s.PreparedB)
        {
            var w = FillF(s.N, s.K);
            var g = new ComputationalGraph();
            g.Metadata["Name"] = "gemm-matrix";
            g.Inputs["x"] = DenseTensor<float>.OfShape(new int[] { s.M, s.N });
            g.Initializers["w"] = w;
            g.Outputs["z"] = DenseTensor<float>.OfShape(new int[] { s.M, s.K });
            g.Nodes.Add(new Node { Name = "mm", Op = OpType.MatMul, Inputs = new[] { "x", "w" }, Outputs = new[] { "z" } });
            g.Options = new ExecutionOptions(OptimizationMode.Speed, tensorOpts);
            g.RefreshLifetimeAnalysis();
            using var scope = Profiler.BeginExecution(true);
            if (!g.Execute(new Dictionary<string, ITensor> { ["x"] = FillF(s.M, s.N) }, true, ExecutionProvider.CPU, new ExecutionOptions(OptimizationMode.Speed, tensorOpts))) throw new InvalidOperationException(s.Name + ": exec failed: " + g.LastErrorMessage);
            y = (Tensor<float>)g.Outputs["z"];
            route = RouteOf(g);
        }
        else
        {
            using var scope = Profiler.BeginExecution(true);
            Profiler.StartNodeProfile(1, OpType.MatMul);
            y = Tensor<float>.MatMul2D(FillF(s.M, s.N), FillF(s.N, s.K), tensorOpts);
            Profiler.StopNodeProfile();
            var snap = Profiler.RouteCountsSnapshot();
            route = snap.Count == 0 ? "none" : string.Join("+", snap.Keys.OrderBy(k => k, StringComparer.Ordinal));
        }
        var ydims = y.Dimensions.ToArray();
        bool shapeOk = ydims.Length == 2 && ydims[0] == s.M && ydims[1] == s.K;
        var zref = Tensor<double>.MatMul(FillD(s.M, s.N), FillD(s.N, s.K), TensorExecutionOptions.Scalar);
        var zf = y.ToArray();
        var zd = zref.ToArray();
        double worst = 0.0;
        if (zf.Length == zd.Length)
        {
            for (int i = 0; i < zf.Length; i++)
            {
                double d = Math.Abs(zf[i] - zd[i]) / Math.Max(1.0, Math.Abs(zd[i]));
                if (d > worst) worst = d;
            }
        }
        else worst = double.PositiveInfinity;
        return new GemmMatrixResult(s.Name, s.M, s.N, s.K, s.PreparedB, route, worst, shapeOk && worst <= Tolerance);
    }
}
