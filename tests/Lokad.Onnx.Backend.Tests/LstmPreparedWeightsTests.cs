namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Covers persistent LSTM transpose preparation: constant W/R weights gain
/// plan-owned transposed clones at preparation, graph execution resolves
/// them, and direct provider calls keep the per-invocation fallback.
/// Byte layouts and execution agreement pin the contract; timing lands in
/// the segmentation/decoder benchmarks.
/// </summary>
public class LstmPreparedWeightsTests
{
    const int Seed = 9171;

    static DenseTensor<float> FillRect(int rows, int cols, Random rnd)
    {
        var t = Tensor<float>.Zeros(rows, cols).ToDenseTensor();
        for (int i = 0; i < t.Length; i++) t.SetValue(i, rnd.NextSingle());
        return t;
    }

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
        graph.Metadata["Name"] = "lstm-prep-test";
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

    static float[] TransposeRef(float[] src, int d, int gh, int k)
    {
        var t = new float[src.Length];
        for (int dir = 0; dir < d; dir++)
            for (int row = 0; row < gh; row++)
                for (int col = 0; col < k; col++)
                    t[(dir * k + col) * gh + row] = src[(dir * gh + row) * k + col];
        return t;
    }

    [Fact]
    public void Prepare_BuildsTransposedClones()
    {
        var rnd = new Random(Seed);
        var x = FillRank3(8, 4, 6, rnd);
        var w = FillRank3(1, 16, 6, rnd);
        var r = FillRank3(1, 16, 4, rnd);
        var graph = BuildLstmGraph(x, w, r, 4, "forward");
        graph.RefreshLifetimeAnalysis();
        Assert.True(graph.Initializers.ContainsKey("lstm-t:w"), "prepared W transpose is missing.");
        Assert.True(graph.Initializers.ContainsKey("lstm-t:r"), "prepared R transpose is missing.");
        var wt = (DenseTensor<float>)graph.Initializers["lstm-t:w"];
        Assert.Equal(new[] { 1, 6, 16 }, wt.Dimensions.ToArray());
        Assert.Equal(TransposeRef(w.ToArray(), 1, 16, 6), wt.ToArray());
        var rt = (DenseTensor<float>)graph.Initializers["lstm-t:r"];
        Assert.Equal(new[] { 1, 4, 16 }, rt.Dimensions.ToArray());
        Assert.Equal(TransposeRef(r.ToArray(), 1, 16, 4), rt.ToArray());
    }

    [Fact]
    public void Execute_PreparedAgreesWithDirect()
    {
        var rnd = new Random(Seed);
        var x = FillRank3(8, 4, 6, rnd);
        var w = FillRank3(1, 16, 6, rnd);
        var r = FillRank3(1, 16, 4, rnd);
        var graph = BuildLstmGraph(x, w, r, 4, "forward");
        var user = new Dictionary<string, ITensor> { ["x"] = x };
        Assert.True(graph.Execute(user, true), graph.LastErrorMessage);
        var viaGraph = ((Tensor<float>)graph.Outputs["y"]).ToArray();
        var direct = CPUExecutionProvider.Lstm(x, w, r, null, null, null, null, null, "forward", null, null, null, null, 4, false, 0, 1, null, null);
        Assert.Equal(OpStatus.Success, direct.Status);
        Assert.Equal(viaGraph, ((Tensor<float>)direct.Outputs[0]).ToArray());
    }

    [Fact]
    public void Execute_BidirectionalPreparedAgreesWithDirect()
    {
        var rnd = new Random(Seed);
        var x = FillRank3(8, 2, 6, rnd);
        var w = FillRank3(2, 16, 6, rnd);
        var r = FillRank3(2, 16, 4, rnd);
        var graph = BuildLstmGraph(x, w, r, 4, "bidirectional");
        graph.RefreshLifetimeAnalysis();
        Assert.True(graph.Initializers.ContainsKey("lstm-t:w"), "prepared bidirectional transpose is missing.");
        var user = new Dictionary<string, ITensor> { ["x"] = x };
        Assert.True(graph.Execute(user, true), graph.LastErrorMessage);
        var viaGraph = ((Tensor<float>)graph.Outputs["y"]).ToArray();
        var direct = CPUExecutionProvider.Lstm(x, w, r, null, null, null, null, null, "bidirectional", null, null, null, null, 4, false, 0, 1, null, null);
        Assert.Equal(OpStatus.Success, direct.Status);
        // Short prepared sequences take the shared MatMul projection while
        // the direct call keeps scalar row dots, so agreement is within
        // float rounding (observed ~6e-8), not bit identity. The transpose
        // itself stays pinned bit-exact by Prepare_BuildsTransposedClones.
        var viaDirect = ((Tensor<float>)direct.Outputs[0]).ToArray();
        Assert.Equal(viaGraph.Length, viaDirect.Length);
        for (int i = 0; i < viaGraph.Length; i++)
        {
            double tol = 1e-6 * (1.0 + System.Math.Abs((double)viaGraph[i]));
            Assert.True(System.Math.Abs((double)viaGraph[i] - viaDirect[i]) <= tol, "index " + i);
        }
    }

    [Fact]
    public void Prepare_SkipsFedWeights()
    {
        var rnd = new Random(Seed);
        var x = FillRank3(8, 4, 6, rnd);
        var w = FillRank3(1, 16, 6, rnd);
        var r = FillRank3(1, 16, 4, rnd);
        var graph = BuildLstmGraph(x, w, r, 4, "forward");
        graph.Initializers.Remove("w");
        graph.Inputs["w"] = w;
        graph.RefreshLifetimeAnalysis();
        Assert.False(graph.Initializers.ContainsKey("lstm-t:w"), "fed weights must not prepare.");
        var user = new Dictionary<string, ITensor> { ["x"] = x, ["w"] = w };
        Assert.True(graph.Execute(user, true), graph.LastErrorMessage);
    }

    [Fact]
    public void Prepare_RebuildsOnReplacement()
    {
        var rnd = new Random(Seed);
        var x = FillRank3(8, 4, 6, rnd);
        var w = FillRank3(1, 16, 6, rnd);
        var r = FillRank3(1, 16, 4, rnd);
        var graph = BuildLstmGraph(x, w, r, 4, "forward");
        graph.RefreshLifetimeAnalysis();
        var w2 = FillRank3(1, 16, 6, new Random(Seed + 1));
        graph.Initializers["w"] = w2;
        graph.RefreshLifetimeAnalysis();
        var wt = (DenseTensor<float>)graph.Initializers["lstm-t:w"];
        Assert.Equal(TransposeRef(w2.ToArray(), 1, 16, 6), wt.ToArray());
        var user = new Dictionary<string, ITensor> { ["x"] = x };
        Assert.True(graph.Execute(user, true), graph.LastErrorMessage);
    }

    [Fact]
    public void Prepare_DropsClonesOnInvalidate()
    {
        var rnd = new Random(Seed);
        var graph = BuildLstmGraph(FillRank3(8, 4, 6, rnd), FillRank3(1, 16, 6, rnd), FillRank3(1, 16, 4, rnd), 4, "forward");
        graph.RefreshLifetimeAnalysis();
        Assert.True(graph.Initializers.ContainsKey("lstm-t:w"), "expected a prepared clone.");
        graph.InvalidatePreparation();
        Assert.False(graph.Initializers.ContainsKey("lstm-t:w"), "clone must drop on invalidate.");
    }
    [SkippableFact]
    public void RealSegmentation_PreparesEveryLstmWeight()
    {
        var graph = ModelFixture.LoadRequiredModel("PyannoteSegmentation", "models", "speaker-diarization-community-1", "onnx", "segmentation", "model.onnx");
        int lstmNodes = 0;
        foreach (var node in graph.Nodes)
        {
            if (node.Op != OpType.LSTM) continue;
            lstmNodes++;
            Assert.True(node.Inputs.Length >= 3, "LSTM node carries W/R inputs.");
            Assert.True(graph.Initializers.ContainsKey("lstm-t:" + node.Inputs[1]), "missing prepared W for " + node.Name);
        }
        Assert.Equal(4, lstmNodes);
    }
}
