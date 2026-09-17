using System;
using System.Collections.Generic;
using Lokad.Onnx;

namespace Lokad.Onnx.Backend.Tests;

// Covers panel-packed LSTM input-weight preparation: constant W gains a
// plan-owned direction-major packed clone at preparation, graph execution
// resolves it for the hoisted XW projection, and direct provider calls keep
// the unpacked fallback. Clone bytes and execution agreement pin the
// contract; timing lands in the decoder benchmarks.
public class LstmPackedWeightsTests
{
    const int Seed = 9171;

    static DenseTensor<float> FillRank3(int d0, int d1, int d2, Random rnd)
    {
        var t = DenseTensor<float>.OfShape(new[] { d0, d1, d2 });
        var span = t.Buffer.Span;
        for (int i = 0; i < span.Length; i++) span[i] = rnd.NextSingle();
        return t;
    }

    static ComputationalGraph BuildLstmGraph(DenseTensor<float> x, DenseTensor<float> w, DenseTensor<float> r, int hidden, string direction)
    {
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "lstm-pack-test";
        graph.Inputs["x"] = x;
        graph.Initializers["w"] = w;
        graph.Initializers["r"] = r;
        int seq = x.Dimensions[0], batch = x.Dimensions[1], dirs = w.Dimensions[0];
        graph.Outputs["y"] = Tensor<float>.Zeros(seq, dirs, batch, hidden).ToDenseTensor();
        graph.Nodes.Add(new Node
        {
            Name = "lstm",
            Op = OpType.LSTM,
            OpTypeName = "LSTM",
            Domain = "",
            Inputs = new[] { "x", "w", "r" },
            Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { ["hidden_size"] = hidden, ["direction"] = direction },
        });
        return graph;
    }

    static float[] PackRef(float[] w, int d, int gh, int k)
    {
        var t = new float[w.Length];
        for (int dir = 0; dir < d; dir++)
        {
            var tmp = new float[k * gh];
            for (int row = 0; row < gh; row++)
                for (int col = 0; col < k; col++)
                    tmp[col * gh + row] = w[(dir * gh + row) * k + col];
            unsafe
            {
                fixed (float* tp = tmp)
                fixed (float* dp = t)
                    MathOps.PackPanelsB(k, gh, tp, dp + dir * k * gh);
            }
        }
        return t;
    }

    [Fact]
    public void Prepare_BuildsPackedClones()
    {
        var rnd = new Random(Seed);
        var x = FillRank3(4, 2, 6, rnd);
        var w = FillRank3(1, 16, 6, rnd);
        var r = FillRank3(1, 16, 4, rnd);
        var graph = BuildLstmGraph(x, w, r, 4, "forward");
        graph.RefreshLifetimeAnalysis();
        Assert.True(graph.Initializers.ContainsKey("lstm-p:w"), "prepared W pack is missing.");
        Assert.False(graph.Initializers.ContainsKey("lstm-p:r"), "R must not gain a packed clone.");
        var wp = (DenseTensor<float>)graph.Initializers["lstm-p:w"];
        Assert.Equal(new[] { 1, 6, 16 }, wp.Dimensions.ToArray());
        Assert.Equal(PackRef(w.ToArray(), 1, 16, 6), wp.ToArray());
    }

    [Fact]
    public void Execute_PreparedPackedAgreesWithDirect()
    {
        var rnd = new Random(Seed);
        var x = FillRank3(4, 2, 6, rnd);
        var w = FillRank3(1, 16, 6, rnd);
        var r = FillRank3(1, 16, 4, rnd);
        var graph = BuildLstmGraph(x, w, r, 4, "forward");
        graph.RefreshLifetimeAnalysis();
        var user = new Dictionary<string, ITensor> { ["x"] = x };
        Assert.True(graph.Execute(user, true), graph.LastErrorMessage);
        var viaGraph = ((Tensor<float>)graph.Outputs["y"]).ToArray();
        var direct = CPUExecutionProvider.Lstm(x, w, r, null, null, null, null, null, "forward", null, null, null, null, 4, false, 0, 1, null, null);
        Assert.Equal(OpStatus.Success, direct.Status);
        // The prepared graph routes XW through the packed kernels while the
        // direct call keeps the unpacked lane, so agreement is within float
        // rounding, not bit identity. Clone bytes stay pinned by
        // Prepare_BuildsPackedClones.
        var viaDirect = ((Tensor<float>)direct.Outputs[0]).ToArray();
        Assert.Equal(viaGraph.Length, viaDirect.Length);
        for (int i = 0; i < viaGraph.Length; i++)
        {
            double tol = 1e-6 * (1.0 + System.Math.Abs((double)viaGraph[i]));
            Assert.True(System.Math.Abs((double)viaGraph[i] - viaDirect[i]) <= tol, "index " + i);
        }
    }

    [Fact]
    public void Prepare_DropsPackOnInvalidate()
    {
        var rnd = new Random(Seed);
        var graph = BuildLstmGraph(FillRank3(4, 2, 6, rnd), FillRank3(1, 16, 6, rnd), FillRank3(1, 16, 4, rnd), 4, "forward");
        graph.RefreshLifetimeAnalysis();
        Assert.True(graph.Initializers.ContainsKey("lstm-p:w"), "expected a prepared pack.");
        graph.InvalidatePreparation();
        Assert.False(graph.Initializers.ContainsKey("lstm-p:w"), "pack must drop on invalidate.");
    }
}