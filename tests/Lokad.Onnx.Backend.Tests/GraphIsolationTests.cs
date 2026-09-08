using System.Threading;

namespace Lokad.Onnx.Backend.Tests;

public class GraphIsolationTests
{
    static OnnxModel ReluModel(OnnxValueInfo input, OnnxValueInfo output)
    {
        var m = new OnnxModel { Name = "m" };
        m.Inputs.Add(input);
        m.Outputs.Add(output);
        m.Nodes.Add(new OnnxNode { Name = "r", OpType = "Relu", Inputs = new string[] { "x" }, Outputs = new string[] { "y" } });
        return m;
    }

    static OnnxValueInfo Vp(string name, int[] dims, string?[]? ps)
    {
        return new OnnxValueInfo { Name = name, ElementType = TensorElementType.Float, Dims = dims, DimParams = ps };
    }

    [Fact]
    public void RepeatedRuns_ChangingShapes()
    {
        var g = Model.Load(ReluModel(
            Vp("x", new int[] { 0, 3 }, new string?[] { "n", null }),
            Vp("y", new int[] { 0, 3 }, new string?[] { "n", null })));
        foreach (int n in new int[] { 5, 2, 4 })
        {
            var data = new float[n, 3];
            for (int i = 0; i < n; i++) for (int j = 0; j < 3; j++) data[i, j] = (float)(i * 3 + j) - 4f;
            var user = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(data) } };
            Assert.True(g.Execute(user, false));
            var y = (Tensor<float>)g.Outputs["y"];
            Assert.Equal(new int[] { n, 3 }, y.Dimensions.ToArray());
            Assert.Equal(Math.Max(0f, data[0, 0]), y.ToArray()[0], 5);
        }
    }

    [Fact]
    public void DirectContext_Equivalence_And_Facade_Untouched()
    {
        var g = Model.Load(ReluModel(Vp("x", new int[] { 2 }, null), Vp("y", new int[] { 2 }, null)));
        var exec = g.CreateExecution(null);
        var user = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f }) } };
        Assert.True(exec.Execute(user, false));
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)exec.Outputs["y"]).ToArray());
        Assert.Equal(new int[] { 2 }, g.Outputs["y"].Dims.ToArray());
        Assert.Throws<InvalidOperationException>(() => g.Outputs["y"].ToArray());
    }

    [Fact]
    public void OutputOwnership_UserMutationInvisible()
    {
        var g = Model.Load(ReluModel(Vp("x", new int[] { 2 }, null), Vp("y", new int[] { 2 }, null)));
        var x = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        Assert.True(g.Execute(new Dictionary<string, ITensor> { { "x", x } }, false));
        x.SetValue(0, 9f);
        x.SetValue(1, 9f);
        Assert.Equal(new float[] { 1f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
    }

    [Fact]
    public void ContextFailure_Reusable()
    {
        var g = Model.Load(ReluModel(Vp("x", new int[] { 2 }, null), Vp("y", new int[] { 2 }, null)));
        var exec = g.CreateExecution(null);
        var bad = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f, 3f }) } };
        Assert.False(exec.Execute(bad, false));
        Assert.False(exec.Outputs.ContainsKey("y"));
        var good = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }) } };
        Assert.True(exec.Execute(good, false));
        Assert.Equal(new float[] { 1f, 2f }, ((Tensor<float>)exec.Outputs["y"]).ToArray());
    }

    static string MnistAsset(string name) => Path.Combine(AppContext.BaseDirectory, name);

    [Fact]
    public void ConcurrentContexts_CorrectOutputs()
    {
        var g = OnnxImport.Load(Path.Combine(MnistAsset("models"), "mnist-8.onnx"))!;
        var uiA = Data.GetInputTensorsFromFileArgs(new[] { Path.Combine(MnistAsset("images"), "mnist4.png::mnist") })!;
        var uiB = Data.GetInputTensorsFromFileArgs(new[] { Path.Combine(MnistAsset("images"), "mnist2.png::mnist") })!;
        Assert.True(g.Execute(uiA, true));
        var refA = ((Tensor<float>)g.Outputs.Values.First()).ToArray();
        Assert.True(g.Execute(uiB, true));
        var refB = ((Tensor<float>)g.Outputs.Values.First()).ToArray();
        var execA = g.CreateExecution(null);
        var execB = g.CreateExecution(null);
        using var barrier = new Barrier(2);
        System.Func<GraphExecution, ITensor[], (bool, float[])> run = (exec, ui) =>
        {
            barrier.SignalAndWait();
            bool ok = exec.Execute(ui, true);
            float[] arr = ok ? ((Tensor<float>)exec.Outputs.Values.First()).ToArray() : Array.Empty<float>();
            return (ok, arr);
        };
        var tA = Task.Run(() => run(execA, uiA));
        var tB = Task.Run(() => run(execB, uiB));
        Task.WaitAll(tA, tB);
        Assert.True(tA.Result.Item1);
        Assert.True(tB.Result.Item1);
        Assert.Equal(refA, tA.Result.Item2);
        Assert.Equal(refB, tB.Result.Item2);
    }

    [Fact]
    public void ConcurrentFacade_RejectsCleanly_And_Recovers()
    {
        var g = OnnxImport.Load(Path.Combine(MnistAsset("models"), "mnist-8.onnx"))!;
        var ui = Data.GetInputTensorsFromFileArgs(new[] { Path.Combine(MnistAsset("images"), "mnist4.png::mnist") })!;
        Assert.True(g.Execute(ui, true));
        var reference = ((Tensor<float>)g.Outputs.Values.First()).ToArray();
        const int rounds = 25;
        var resultsA = new bool[rounds];
        var resultsB = new bool[rounds];
        using var barrier = new Barrier(2);
        var tA = Task.Run(() => { for (int i = 0; i < rounds; i++) { barrier.SignalAndWait(); resultsA[i] = g.Execute(ui, true); } });
        var tB = Task.Run(() => { for (int i = 0; i < rounds; i++) { barrier.SignalAndWait(); resultsB[i] = g.Execute(ui, true); } });
        Task.WaitAll(tA, tB);
        int rejected = resultsA.Count(r => !r) + resultsB.Count(r => !r);
        int succeeded = resultsA.Count(r => r) + resultsB.Count(r => r);
        Assert.True(rejected >= 1, "expected at least one clean rejection under hammer");
        Assert.True(succeeded >= 1, "expected at least one success under hammer");
        Assert.True(g.Execute(ui, true));
        Assert.Equal(reference, ((Tensor<float>)g.Outputs.Values.First()).ToArray());
    }
}
