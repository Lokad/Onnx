namespace Lokad.Onnx.Backend.Tests;

[Collection("SequentialLogSink")]
public class PublicPreparationTests
{
    static ComputationalGraph Graph(bool context)
    {
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "public-preparation";
        graph.Inputs["x"] = DenseTensor<float>.OfShape(2);
        graph.Outputs["y"] = DenseTensor<float>.OfShape(2);
        graph.IntermediateOutputs["h"] = null;
        graph.Nodes.Add(new Node { Name = "first", Op = OpType.Relu, Inputs = ["x"], Outputs = ["h"] });
        graph.Nodes.Add(new Node { Name = "last", Op = OpType.Identity, Inputs = ["h"], Outputs = ["y"] });
        return context ? graph.CreateExecution(ExecutionOptions.Scalar) : graph;
    }

    [Theory]
    [InlineData(false, false, false)] [InlineData(false, false, true)]
    [InlineData(false, true, false)] [InlineData(false, true, true)]
    [InlineData(true, false, false)] [InlineData(true, false, true)]
    [InlineData(true, true, false)] [InlineData(true, true, true)]
    public void PreparationDuringActualExecutionIsRefusedWithoutChangingState(bool context, bool refresh, bool otherThread)
    {
        var graph = Graph(context);
        var inputs = new ITensor[] { DenseTensor<float>.OfValues([-1, 2]) };
        Assert.True(graph.Execute(inputs, false));
        var held = (Tensor<float>)graph.Outputs["y"];
        var lifetime = graph.LastUseIndex;
        var options = graph.Options;
        Exception? rejection = null;
        int callbacks = 0;
        var previousSink = Log.Sink;
        var previousLevel = Log.MinLevel;
        try
        {
            Log.MinLevel = LogLevel.Debug;
            Log.Sink = (_, message) =>
            {
                if (!message.StartsWith("Executing node ") || Interlocked.Exchange(ref callbacks, 1) != 0) return;
                void Attempt()
                {
                    rejection = Record.Exception(() => { if (refresh) graph.RefreshLifetimeAnalysis(); else graph.Prepare(); });
                }
                if (otherThread)
                {
                    var task = Task.Factory.StartNew(Attempt, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
                    Assert.True(task.Wait(TimeSpan.FromSeconds(10)), "Preparation must reject an active owner without waiting for its run.");
                }
                else Attempt();
                Assert.Same(lifetime, graph.LastUseIndex);
                Assert.Same(options, graph.Options);
                Assert.Null(graph.LastErrorMessage);
            };
            bool ok = false;
            var executionError = Record.Exception(() => ok = graph.Execute(inputs, false));
            Assert.Null(executionError);
            Assert.True(ok, graph.LastErrorMessage);
            Assert.Equal(1, callbacks);
            Assert.IsType<InvalidOperationException>(rejection);
            Assert.Equal(new float[] { 0, 2 }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
            Assert.Equal(new float[] { 0, 2 }, held.ToArray());
        }
        finally { Log.Sink = previousSink; Log.MinLevel = previousLevel; }
        graph.RefreshLifetimeAnalysis();
        graph.Prepare();
        Assert.True(graph.Execute(inputs, false), graph.LastErrorMessage);
    }
}
