namespace Lokad.Onnx.Backend.Tests;

public class SingleNodeProfilingTests
{
    static ComputationalGraph Graph(bool explicitContext, bool reshape = false)
    {
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "single-node-profile";
        graph.Inputs["x"] = DenseTensor<float>.OfShape(2);
        graph.Outputs["y"] = DenseTensor<float>.OfShape(2);
        if (reshape) graph.Initializers["shape"] = DenseTensor<long>.OfValues(new long[] { 3 });
        graph.Nodes.Add(new Node { Name = "r", Op = reshape ? OpType.Reshape : OpType.Relu,
            Inputs = reshape ? new[] { "x", "shape" } : new[] { "x" }, Outputs = new[] { "y" } });
        graph.Prepare();
        return explicitContext ? graph.CreateExecution(ExecutionOptions.Scalar) : graph;
    }

    static bool Execute(ComputationalGraph graph) => graph.ExecuteNode(
        graph.Nodes[0].Inputs.Length == 0 ? new Dictionary<string, ITensor>() :
            new Dictionary<string, ITensor> { ["x"] = DenseTensor<float>.OfValues(new float[] { -1, 2 }) },
        "r", true, ExecutionProvider.CPU, ExecutionOptions.Scalar);

    static void CheckProfile(ComputationalGraph graph, bool wall, OpType op)
    {
        if (wall)
        {
            var node = Assert.Single(graph.LastWallProfile!);
            Assert.Equal(0, node.NodeId);
            Assert.Equal(op, node.Op);
            Assert.True(node.StartTicks > 0 && node.EndTicks >= node.StartTicks);
            Assert.Empty(graph.LastProfile!);
        }
        else
        {
            var node = Assert.Single(graph.LastProfile!);
            Assert.Equal(0, node.NodeId);
            Assert.Equal(op, node.Op);
            Assert.NotEmpty(node.OpsProfile);
            Assert.All(node.OpsProfile, stage => Assert.True(stage.Time >= TimeSpan.Zero));
            Assert.True(node.OpsProfile.Sum(stage => stage.Time.Ticks) <= graph.LastRunTime.Ticks);
            Assert.Empty(graph.LastWallProfile!);
        }
    }

    static void CheckRestoredParent(ProfilerContext parent, bool wall)
    {
        Profiler.StartOpStage(OpStage.Copy);
        Profiler.StopNodeProfile();
        if (wall)
        {
            var node = Assert.Single(parent.Wall);
            Assert.Equal(1000, node.NodeId);
            Assert.True(node.EndTicks >= node.StartTicks);
            Assert.Empty(parent.Profile);
        }
        else
        {
            var node = Assert.Single(parent.Profile);
            Assert.Equal(1000, node.NodeId);
            Assert.Contains(node.OpsProfile, stage => stage.Stage == OpStage.Copy);
            Assert.DoesNotContain(node.OpsProfile, stage => stage.Stage == OpStage.Math);
            Assert.Empty(parent.Wall);
        }
    }

    [Theory]
    [InlineData(false, false)] [InlineData(false, true)]
    [InlineData(true, false)] [InlineData(true, true)]
    public void ProfilesOneNodeWithoutChangingOutput(bool explicitContext, bool wall)
    {
        var graph = Graph(explicitContext);
        using var parent = wall ? Profiler.BeginWallExecution() : Profiler.BeginExecution(true);
        parent.StartNodeProfile(1000, OpType.Identity);
        Assert.True(Execute(graph), graph.LastErrorMessage);
        Assert.Equal(new float[] { 0, 2 }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
        CheckProfile(graph, wall, OpType.Relu);
        CheckRestoredParent(parent, wall);
    }

    [Theory]
    [InlineData(false, false)] [InlineData(false, true)]
    [InlineData(true, false)] [InlineData(true, true)]
    public void FailedNodeRestoresScopeAndCorrectedRetryPublishesFreshProfile(bool explicitContext, bool wall)
    {
        var graph = Graph(explicitContext, reshape: true);
        using var parent = wall ? Profiler.BeginWallExecution() : Profiler.BeginExecution(true);
        parent.StartNodeProfile(1000, OpType.Identity);
        Assert.False(Execute(graph));
        Assert.NotNull(graph.LastErrorMessage);
        Assert.DoesNotContain("Stack empty", graph.LastErrorMessage);
        Assert.Empty(graph.Outputs);
        Assert.Null(graph.LastProfile);
        Assert.Null(graph.LastWallProfile);
        graph.Initializers["shape"] = DenseTensor<long>.OfValues(new long[] { 2 });
        Assert.True(Execute(graph), graph.LastErrorMessage);
        Assert.Null(graph.LastErrorMessage);
        Assert.Equal(new float[] { -1, 2 }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
        CheckProfile(graph, wall, OpType.Reshape);
        CheckRestoredParent(parent, wall);
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void DisabledProfilingKeepsSuccessfulProfilesEmpty(bool explicitContext)
    {
        var graph = Graph(explicitContext);
        using var scope = Profiler.BeginExecution(false);
        Assert.True(Execute(graph), graph.LastErrorMessage);
        Assert.NotNull(graph.LastProfile);
        Assert.NotNull(graph.LastWallProfile);
        Assert.Empty(graph.LastProfile);
        Assert.Empty(graph.LastWallProfile);
    }

    sealed class FailingConstant(Exception failure) : DenseTensor<float>(new float[] { 1, 2 }, new[] { 2 })
    {
        public override DenseTensor<float> ToDenseTensor() => throw failure;
    }

    [Theory]
    [InlineData(false, false)] [InlineData(false, true)]
    [InlineData(true, false)] [InlineData(true, true)]
    public void FatalNodeFailurePropagatesAndRestoresParent(bool explicitContext, bool wall)
    {
        // A controlled exception, not actual memory exhaustion.
        var failure = new OutOfMemoryException("controlled profile failure");
        var graph = Graph(false);
        graph.Nodes[0] = new Node { Name = "r", Op = OpType.Constant, Inputs = [], Outputs = ["y"],
            Attributes = new() { ["value"] = new FailingConstant(failure) } };
        graph.Prepare();
        if (explicitContext) graph = graph.CreateExecution(ExecutionOptions.Scalar);
        using var parent = wall ? Profiler.BeginWallExecution() : Profiler.BeginExecution(true);
        parent.StartNodeProfile(1000, OpType.Identity);
        Assert.Same(failure, Record.Exception(() => Execute(graph)));
        CheckRestoredParent(parent, wall);
    }
}
