namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Guards the P16 Milestone 1 contract: the preparation pass builds
/// panel-packed clones of eligible MatMul weight initializers, skips every
/// ineligible shape and consumer, leaves source bytes untouched, and drops
/// clones on invalidation. Dispatch is untouched in this milestone, so packed
/// graphs must execute exactly like unpacked ones. All assertions use the
/// public surface: initializer keys, clone bytes, and executed values.
/// </summary>
public class PackedWeightsTests
{
    const int Seed = 4242;

    static DenseTensor<float> FillRect(int rows, int cols, Random rnd)
    {
        var t = Tensor<float>.Zeros(rows, cols).ToDenseTensor();
        for (int i = 0; i < t.Length; i++) t.SetValue(i, rnd.NextSingle());
        return t;
    }

    static ComputationalGraph BuildGraph(DenseTensor<float> xFeed, DenseTensor<float> w, string wName)
    {
        return BuildGraphWithConsumer(xFeed, w, wName, false);
    }

    static ComputationalGraph BuildGraphWithConsumer(DenseTensor<float> xFeed, DenseTensor<float> w, string wName, bool extraConsumer)
    {
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "packed-test";
        graph.Inputs["x"] = xFeed;
        graph.Initializers[wName] = w;
        int m = xFeed.Dimensions[0];
        int k = w.Dimensions[1];
        graph.Outputs["z"] = Tensor<float>.Zeros(m, k).ToDenseTensor();
        graph.Nodes.Add(new Node
        {
            Name = "mm",
            Op = OpType.MatMul,
            OpTypeName = "MatMul",
            Domain = "",
            Inputs = new[] { "x", wName },
            Outputs = new[] { "z" },
        });
        if (extraConsumer)
        {
            graph.Outputs["z2"] = Tensor<float>.Zeros(m, k).ToDenseTensor();
            graph.Nodes.Add(new Node
            {
                Name = "add",
                Op = OpType.Add,
                OpTypeName = "Add",
                Domain = "",
                Inputs = new[] { "z", wName },
                Outputs = new[] { "z2" },
            });
        }
        return graph;
    }

    static float[] ReferenceProduct(DenseTensor<float> a, DenseTensor<float> b)
    {
        int m = a.Dimensions[0], n = a.Dimensions[1], k = b.Dimensions[1];
        var c = new float[m * k];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++)
            {
                float acc = 0f;
                for (int l = 0; l < n; l++) acc += a.GetValue(i * n + l) * b.GetValue(l * k + j);
                c[i * k + j] = acc;
            }
        return c;
    }

    static void AgreesWithReference(DenseTensor<float> a, DenseTensor<float> b, Tensor<float> got, string what)
    {
        var expected = ReferenceProduct(a, b);
        var actual = got.ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-3f * Math.Max(1f, Math.Abs(expected[i])), what + " mismatch at " + i);
    }

    [Fact]
    public void PackPass_PacksEligibleWeight()
    {
        var rnd = new Random(Seed);
        var x = FillRect(8, 24, rnd);
        var w = FillRect(24, 44, rnd);
        var before = w.ToArray();
        var graph = BuildGraph(x, w, "w");
        graph.RefreshLifetimeAnalysis();
        Assert.True(graph.Initializers.ContainsKey("packed:w"), "packed clone is missing.");
        var packed = (DenseTensor<float>)graph.Initializers["packed:w"];
        Assert.Equal(new[] { 24, 44 }, packed.Dimensions.ToArray());
        Assert.Equal(before, w.ToArray());
        var expect = new float[24 * 44];
        unsafe
        {
            using var sh = w.Buffer.Pin();
            using var ph = new Memory<float>(expect).Pin();
            MathOps.PackPanelsB(24, 44, (float*)sh.Pointer, (float*)ph.Pointer);
        }
        Assert.Equal(expect, packed.ToArray());
    }

    [Fact]
    public void PackPass_SkipsIneligibleConsumersAndShapes()
    {
        var rnd = new Random(Seed);
        var shared = BuildGraphWithConsumer(FillRect(8, 24, rnd), FillRect(24, 44, rnd), "w", true);
        shared.RefreshLifetimeAnalysis();
        Assert.False(shared.Initializers.ContainsKey("packed:w"), "shared weight must not pack.");
        var wide = BuildGraph(FillRect(8, 2560, rnd), FillRect(2560, 8, rnd), "w");
        wide.RefreshLifetimeAnalysis();
        Assert.False(wide.Initializers.ContainsKey("packed:w"), "wide-axis weight must not pack.");
        var dbl = new ComputationalGraph();
        dbl.Metadata["Name"] = "packed-test";
        var xd = FillRect(8, 24, rnd);
        dbl.Inputs["x"] = xd;
        dbl.Initializers["w"] = DenseTensor<double>.OfValues(new double[,] { { 1.0 } });
        dbl.Outputs["z"] = Tensor<float>.Zeros(8, 1).ToDenseTensor();
        dbl.Nodes.Add(new Node { Name = "mm", Op = OpType.MatMul, OpTypeName = "MatMul", Domain = "", Inputs = new[] { "x", "w" }, Outputs = new[] { "z" } });
        dbl.RefreshLifetimeAnalysis();
        Assert.False(dbl.Initializers.ContainsKey("packed:w"), "non-float weight must not pack.");
    }

    [Fact]
    public void PackPass_SkipsGraphOutputWeight()
    {
        var rnd = new Random(Seed);
        var x = FillRect(8, 24, rnd);
        var w = FillRect(24, 44, rnd);
        var graph = BuildGraph(x, w, "w");
        graph.Outputs["w"] = w;
        graph.RefreshLifetimeAnalysis();
        Assert.False(graph.Initializers.ContainsKey("packed:w"), "graph-output weight must not pack.");
    }

    [Fact]
    public void PackPass_ExecuteAgreesWithReference()
    {
        var rnd = new Random(Seed);
        var x = FillRect(8, 24, rnd);
        var w = FillRect(24, 44, rnd);
        var graph = BuildGraph(x, w, "w");
        var user = new Dictionary<string, ITensor> { ["x"] = x };
        Assert.True(graph.Execute(user, true), graph.LastErrorMessage + " / node=" + graph.LastFailedNodeName);
        Assert.True(graph.Initializers.ContainsKey("packed:w"), "expected a live clone after Execute.");
        AgreesWithReference(x, w, (Tensor<float>)graph.Outputs["z"], "packed-graph");
    }

    [Fact]
    public void PackPass_InvalidateDropsClonesAndRebuilds()
    {
        var rnd = new Random(Seed);
        var x = FillRect(8, 24, rnd);
        var w = FillRect(24, 44, rnd);
        var graph = BuildGraph(x, w, "w");
        var user = new Dictionary<string, ITensor> { ["x"] = x };
        Assert.True(graph.Execute(user, true), graph.LastErrorMessage);
        Assert.True(graph.Initializers.ContainsKey("packed:w"), "expected a live clone after Execute.");
        graph.InvalidatePreparation();
        Assert.False(graph.Initializers.ContainsKey("packed:w"), "clone must drop on invalidate.");
        Assert.True(graph.Execute(user, true), graph.LastErrorMessage + " after invalidate");
        Assert.True(graph.Initializers.ContainsKey("packed:w"), "clone must rebuild on next Prepare.");
        AgreesWithReference(x, w, (Tensor<float>)graph.Outputs["z"], "rebuilt");
    }

    static ComputationalGraph BuildRank3Graph(DenseTensor<float> xFeed, DenseTensor<float> w, string wName)
    {
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "packed-test-3d";
        graph.Inputs["x"] = xFeed;
        graph.Initializers[wName] = w;
        int b = xFeed.Dimensions[0], m = xFeed.Dimensions[1], k = w.Dimensions[1];
        graph.Outputs["z"] = Tensor<float>.Zeros(b, m, k).ToDenseTensor();
        graph.Nodes.Add(new Node
        {
            Name = "mm",
            Op = OpType.MatMul,
            OpTypeName = "MatMul",
            Domain = "",
            Inputs = new[] { "x", wName },
            Outputs = new[] { "z" },
        });
        return graph;
    }

    static void AgreesBatched(DenseTensor<float> a, DenseTensor<float> b, Tensor<float> got, string what)
    {
        int batch = a.Dimensions[0], m = a.Dimensions[1], n = a.Dimensions[2], k = b.Dimensions[1];
        var actual = got.ToArray();
        Assert.Equal(batch * m * k, actual.Length);
        for (int bb = 0; bb < batch; bb++)
            for (int i = 0; i < m; i++)
                for (int j = 0; j < k; j++)
                {
                    float acc = 0f;
                    for (int l = 0; l < n; l++) acc += a.GetValue(bb * m * n + i * n + l) * b.GetValue(l * k + j);
                    float gotv = actual[bb * m * k + i * k + j];
                    Assert.True(Math.Abs(acc - gotv) <= 1e-3f * Math.Max(1f, Math.Abs(acc)), what + " mismatch at " + bb + "," + i + "," + j);
                }
    }

    static DenseTensor<float> Fill3(int b, int rows, int cols, Random rnd)
    {
        var t = Tensor<float>.Zeros(b, rows, cols).ToDenseTensor();
        for (int i = 0; i < t.Length; i++) t.SetValue(i, rnd.NextSingle());
        return t;
    }

    [Fact]
    public void PackedDispatch_AgreesOnE5Shapes()
    {
        var rnd = new Random(Seed);
        foreach (var shape in new[] { new[] { 8, 24, 44 }, new[] { 8, 40, 44 }, new[] { 12, 24, 20 } })
        {
            var x = FillRect(shape[0], shape[1], rnd);
            var w = FillRect(shape[1], shape[2], rnd);
            var graph = BuildGraph(x, w, "w");
            var user = new Dictionary<string, ITensor> { ["x"] = x };
            Assert.True(graph.Execute(user, true), graph.LastErrorMessage + " / node=" + graph.LastFailedNodeName);
            Assert.True(graph.Initializers.ContainsKey("packed:w"), "expected a live clone.");
            AgreesWithReference(x, w, (Tensor<float>)graph.Outputs["z"], "packed-2d-" + shape[0] + "x" + shape[1] + "x" + shape[2]);
        }
    }

    [Fact]
    public void PackedDispatch_AgreesOnRank3()
    {
        var rnd = new Random(Seed);
        var x = Fill3(1, 8, 24, rnd);
        var w = FillRect(24, 44, rnd);
        var graph = BuildRank3Graph(x, w, "w");
        var user = new Dictionary<string, ITensor> { ["x"] = x };
        Assert.True(graph.Execute(user, true), graph.LastErrorMessage + " / node=" + graph.LastFailedNodeName);
        Assert.True(graph.Initializers.ContainsKey("packed:w"), "expected a live clone.");
        AgreesBatched(x, w, (Tensor<float>)graph.Outputs["z"], "packed-3d");
    }

    [Fact]
    public void PackedDispatch_ScalarFallbackStaysExact()
    {
        var rnd = new Random(Seed);
        var x = FillRect(8, 24, rnd);
        var w = FillRect(24, 44, rnd);
        var graph = BuildGraph(x, w, "w");
        graph.Options = ExecutionOptions.Scalar;
        var user = new Dictionary<string, ITensor> { ["x"] = x };
        Assert.True(graph.Execute(user, true), graph.LastErrorMessage + " / node=" + graph.LastFailedNodeName);
        Assert.True(graph.Initializers.ContainsKey("packed:w"), "clone still builds under Scalar options.");
        AgreesWithReference(x, w, (Tensor<float>)graph.Outputs["z"], "scalar-fallback");
    }

    [Fact]
    public void PackedDispatch_OddMFallsBackExactly()
    {
        var rnd = new Random(Seed);
        var x = FillRect(7, 24, rnd);
        var w = FillRect(24, 44, rnd);
        var graph = BuildGraph(x, w, "w");
        var user = new Dictionary<string, ITensor> { ["x"] = x };
        Assert.True(graph.Execute(user, true), graph.LastErrorMessage + " / node=" + graph.LastFailedNodeName);
        AgreesWithReference(x, w, (Tensor<float>)graph.Outputs["z"], "odd-m-fallback");
    }

    [Fact]
    public void PackedDispatch_StaleSourceFallsBackExactly()
    {
        var rnd = new Random(Seed);
        var x = FillRect(8, 24, rnd);
        var w = FillRect(24, 44, rnd);
        var graph = BuildGraph(x, w, "w");
        var user = new Dictionary<string, ITensor> { ["x"] = x };
        Assert.True(graph.Execute(user, true), graph.LastErrorMessage);
        var w2 = FillRect(24, 44, rnd);
        graph.Initializers["w"] = w2;
        Assert.True(graph.Execute(user, true), graph.LastErrorMessage + " after replace");
        AgreesWithReference(x, w2, (Tensor<float>)graph.Outputs["z"], "stale-source");
    }

    [Fact]
    public void PackedDispatch_SharedWeightServesBoth()
    {
        var rnd = new Random(Seed);
        var x = FillRect(8, 24, rnd);
        var w = FillRect(24, 44, rnd);
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "packed-test-shared";
        graph.Inputs["x"] = x;
        graph.Initializers["w"] = w;
        graph.Outputs["z1"] = Tensor<float>.Zeros(8, 44).ToDenseTensor();
        graph.Outputs["z2"] = Tensor<float>.Zeros(8, 44).ToDenseTensor();
        graph.Nodes.Add(new Node { Name = "m1", Op = OpType.MatMul, OpTypeName = "MatMul", Domain = "", Inputs = new[] { "x", "w" }, Outputs = new[] { "z1" } });
        graph.Nodes.Add(new Node { Name = "m2", Op = OpType.MatMul, OpTypeName = "MatMul", Domain = "", Inputs = new[] { "x", "w" }, Outputs = new[] { "z2" } });
        var user = new Dictionary<string, ITensor> { ["x"] = x };
        Assert.True(graph.Execute(user, true), graph.LastErrorMessage + " / node=" + graph.LastFailedNodeName);
        int clones = 0;
        foreach (var key in graph.Initializers.Keys) if (key.StartsWith("packed:")) clones++;
        Assert.Equal(1, clones);
        AgreesWithReference(x, w, (Tensor<float>)graph.Outputs["z1"], "shared-1");
        AgreesWithReference(x, w, (Tensor<float>)graph.Outputs["z2"], "shared-2");
    }
    static ComputationalGraph BuildGemmGraph(DenseTensor<float> xFeed, DenseTensor<float> w, string wName, int transB)
    {
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "packed-test-gemm";
        graph.Inputs["x"] = xFeed;
        graph.Initializers[wName] = w;
        int m = xFeed.Dimensions[0];
        int k = transB == 1 ? w.Dimensions[0] : w.Dimensions[1];
        graph.Outputs["z"] = Tensor<float>.Zeros(m, k).ToDenseTensor();
        var node = new Node
        {
            Name = "gm",
            Op = OpType.Gemm,
            OpTypeName = "Gemm",
            Domain = "",
            Inputs = new[] { "x", wName },
            Outputs = new[] { "z" },
        };
        node.Attributes = new Dictionary<string, object> { ["transB"] = transB };
        graph.Nodes.Add(node);
        return graph;
    }

    [Fact]
    public void PackPass_PacksPlainGemmWeight()
    {
        var rnd = new Random(Seed);
        var x = FillRect(4, 24, rnd);
        var w = FillRect(24, 44, rnd);
        var graph = BuildGemmGraph(x, w, "w", 0);
        graph.RefreshLifetimeAnalysis();
        Assert.True(graph.Initializers.ContainsKey("packed:w"), "packed clone is missing for plain Gemm.");
        var user = new Dictionary<string, ITensor> { ["x"] = x };
        Assert.True(graph.Execute(user, true), graph.LastErrorMessage + " / node=" + graph.LastFailedNodeName);
        AgreesWithReference(x, w, (Tensor<float>)graph.Outputs["z"], "packed-gemm");
    }

    [Fact]
    public void PackPass_SkipsTransposedGemmWeight()
    {
        var rnd = new Random(Seed);
        var x = FillRect(4, 44, rnd);
        var w = FillRect(24, 44, rnd);
        var graph = BuildGemmGraph(x, w, "w", 1);
        graph.RefreshLifetimeAnalysis();
        Assert.False(graph.Initializers.ContainsKey("packed:w"), "transposed Gemm weight must not pack.");
        var user = new Dictionary<string, ITensor> { ["x"] = x };
        Assert.True(graph.Execute(user, true), graph.LastErrorMessage + " / node=" + graph.LastFailedNodeName);
        var wsp = w.Buffer.Span;
        var tw = new float[44 * 24];
        for (int j = 0; j < 24; j++) for (int l = 0; l < 44; l++) tw[l * 24 + j] = wsp[j * 44 + l];
        var a = x.ToArray();
        var exp2 = new float[4 * 24];
        for (int i = 0; i < 4; i++) for (int j = 0; j < 24; j++) { float acc = 0f; for (int l = 0; l < 44; l++) acc += a[i * 44 + l] * tw[l * 24 + j]; exp2[i * 24 + j] = acc; }
        var actual = ((Tensor<float>)graph.Outputs["z"]).ToArray();
        Assert.Equal(exp2.Length, actual.Length);
        for (int i = 0; i < exp2.Length; i++) Assert.True(Math.Abs(exp2[i] - actual[i]) <= 1e-3f * Math.Max(1f, Math.Abs(exp2[i])), "transposed-gemm mismatch at " + i);
    }

    [Fact]
    public void PackPass_SharedMatMulGemmWeightPacks()
    {
        var rnd = new Random(Seed);
        var x = FillRect(4, 24, rnd);
        var w = FillRect(24, 44, rnd);
        var graph = BuildGraph(x, w, "w");
        graph.Outputs["z2"] = Tensor<float>.Zeros(4, 44).ToDenseTensor();
        var gnode = new Node { Name = "gm", Op = OpType.Gemm, OpTypeName = "Gemm", Domain = "", Inputs = new[] { "x", "w" }, Outputs = new[] { "z2" } };
        gnode.Attributes = new Dictionary<string, object> { ["transB"] = 0 };
        graph.Nodes.Add(gnode);
        graph.RefreshLifetimeAnalysis();
        Assert.True(graph.Initializers.ContainsKey("packed:w"), "shared MatMul+Gemm weight must pack.");
        var user = new Dictionary<string, ITensor> { ["x"] = x };
        Assert.True(graph.Execute(user, true), graph.LastErrorMessage + " / node=" + graph.LastFailedNodeName);
        AgreesWithReference(x, w, (Tensor<float>)graph.Outputs["z"], "shared-mm");
        AgreesWithReference(x, w, (Tensor<float>)graph.Outputs["z2"], "shared-gm");
    }
}
