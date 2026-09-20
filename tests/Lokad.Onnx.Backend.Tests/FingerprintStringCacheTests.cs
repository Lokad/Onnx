using System.Reflection;

namespace Lokad.Onnx.Backend.Tests;

public class FingerprintStringCacheTests
{
    static long Fingerprint(ComputationalGraph graph, bool cached)
    {
        bool saved = graph.CacheFingerprintStrings;
        graph.CacheFingerprintStrings = cached;
        try
        {
            var method = typeof(ComputationalGraph).GetMethod("ComputeStructureFingerprint",
                BindingFlags.Instance | BindingFlags.NonPublic, null, Type.EmptyTypes, null)!;
            return method.CreateDelegate<Func<long>>(graph)();
        }
        finally { graph.CacheFingerprintStrings = saved; }
    }

    static void Agree(ComputationalGraph graph) =>
        Assert.Equal(Fingerprint(graph, false), Fingerprint(graph, true));

    static ComputationalGraph Chain() => Chain(true);

    static ComputationalGraph Chain(bool cached)
    {
        var graph = new ComputationalGraph { CacheFingerprintStrings = cached };
        graph.Metadata["Name"] = "fingerprint-test";
        graph.Inputs["x"] = DenseTensor<float>.OfShape(2);
        graph.Outputs["y"] = DenseTensor<float>.OfShape(2);
        graph.OutputDescs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = [2] });
        graph.Nodes.Add(new Node { Name = "first", Op = OpType.Relu, Inputs = ["x"], Outputs = ["t"] });
        graph.Nodes.Add(new Node { Name = "second", Op = OpType.Relu, Inputs = ["t"], Outputs = ["y"] });
        graph.IntermediateOutputs["t"] = null;
        graph.Prepare();
        return graph;
    }

    static Dictionary<string, ITensor> Feed() => Feed(-1, 2);

    static Dictionary<string, ITensor> Feed(float first, float second) =>
        new() { ["x"] = DenseTensor<float>.OfValues([first, second]) };

    [Fact]
    public void UnicodeNullAndEqualValueNames_PreserveOriginalHash()
    {
        var graph = Chain();
        var snapshot = graph.FingerprintStrings;
        string?[] names = [null, "", "\0", "a\0b", "\ud800", "\udfff", "\ud83d\ude80", "é", new string('x', 257)];
        foreach (var name in names)
        {
            var node = graph.Nodes[0];
            node.Name = name!; node.Domain = name; node.OpTypeName = name;
            graph.Nodes[0] = node;
            Agree(graph);
            node.Name = name is null ? null! : new string(name.ToCharArray());
            graph.Nodes[0] = node;
            Agree(graph);
        }
        Assert.Same(snapshot, graph.FingerprintStrings);
    }

    [Fact]
    public void ChangedIncomingHash_DoesNotReuseSameLaterStrings()
    {
        var graph = Chain();
        long original = Fingerprint(graph, true);
        var node = graph.Nodes[0]; node.Op = OpType.Neg; graph.Nodes[0] = node;
        Agree(graph);
        Assert.NotEqual(original, Fingerprint(graph, true));
        node.Inputs = ["new-input", "another"]; node.Outputs = null!; graph.Nodes[0] = node;
        Agree(graph);
        node.Inputs = null!; node.Outputs = ["new-output"]; graph.Nodes[0] = node;
        Agree(graph);
    }

    [Fact]
    public void UnpreparedAndDisabledGraphs_UseOriginalFallback()
    {
        var graph = Chain(false);
        Assert.Null(graph.FingerprintStrings);
        Agree(graph);
        graph.CacheFingerprintStrings = true;
        Assert.True(graph.Execute(Feed(), false));
        Assert.Null(graph.FingerprintStrings); // Enabling does not mutate snapshots during a check.
        graph.Prepare();
        Assert.NotEmpty(graph.FingerprintStrings!);
        Agree(graph);
        graph.CacheFingerprintStrings = false;
        graph.Prepare();
        Assert.Null(graph.FingerprintStrings);
    }

    [Fact]
    public void ChecksDoNotGrowOrRewritePreparedSnapshot()
    {
        var graph = Chain();
        var snapshot = graph.FingerprintStrings!;
        var original = snapshot.ToArray();
        for (int i = 0; i < 100; i++)
        {
            graph.Nodes.Add(new Node { Name = "later-" + i, Inputs = ["x"], Outputs = ["later-output-" + i] });
            graph.Inputs["input-" + i] = DenseTensor<float>.OfShape(1);
            graph.Initializers["weight-" + i] = DenseTensor<float>.OfShape(1);
            graph.InputDescs.Add(new OnnxValueInfo { Name = "input-" + i });
            graph.OutputDescs.Add(new OnnxValueInfo { Name = "later-output-" + i });
            Agree(graph);
        }
        Assert.Same(snapshot, graph.FingerprintStrings);
        Assert.Equal(original, snapshot);
        graph.Nodes.Reverse(); Agree(graph);
        graph.Nodes.Clear(); Agree(graph);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SameCountRewire_RefreshesLifetimesAndPreservesHeldOutputs(bool explicitContext)
    {
        var owner = Chain();
        var graph = explicitContext ? owner.CreateExecution(ExecutionOptions.Memory) : owner;
        Assert.True(graph.Execute(Feed(), false, ExecutionProvider.CPU, ExecutionOptions.Memory));
        var held = (Tensor<float>)graph.Outputs["y"];
        var oldLifetimes = graph.LastUseIndex;
        var node = graph.Nodes[1]; node.Inputs = ["x"]; node.Op = OpType.Neg; graph.Nodes[1] = node;
        graph.Reset();
        Assert.True(graph.Execute(Feed(), false, ExecutionProvider.CPU, ExecutionOptions.Memory), graph.LastErrorMessage);
        Assert.NotSame(oldLifetimes, graph.LastUseIndex);
        Assert.Equal(0, graph.LastUseIndex["t"]);
        Assert.Equal(new[] { 1f, -2f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
        Assert.Equal(new[] { 0f, 2f }, held.ToArray());
        Assert.False(graph.Execute(new Dictionary<string, ITensor>(), false));
        Assert.True(graph.Execute(Feed(-3, 4), false));
        Assert.Equal(new[] { 0f, 2f }, held.ToArray());
        Agree(graph);
    }

    [Fact]
    public void OutputDeclarationChanges_AreDetectedAfterReset()
    {
        var graph = Chain();
        Assert.True(graph.Execute(Feed(), false));
        var old = graph.LastUseIndex;
        graph.OutputDescs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = [2] });
        graph.Reset();
        Assert.True(graph.Execute(Feed(), false), graph.LastErrorMessage);
        Assert.NotSame(old, graph.LastUseIndex);
        Assert.Equal(new[] { -1f, 2f }, ((Tensor<float>)graph.Outputs["x"]).ToArray());
        Agree(graph);
    }

    [Fact]
    public void NestedSharedGraphChangesAndCycles_AreNotHidden()
    {
        var child = Chain();
        var parent = new ComputationalGraph { CacheFingerprintStrings = true };
        parent.Nodes.Add(new Node { Name = "branches", Inputs = [], Outputs = [], Attributes = new() { ["then"] = child, ["else"] = child } });
        parent.Prepare();
        long original = Fingerprint(parent, true);
        child.Outputs["new-read"] = DenseTensor<float>.OfShape(1);
        Agree(parent); Assert.NotEqual(original, Fingerprint(parent, true));
        child.Nodes[0].Inputs[0] = "changed-capture";
        Agree(parent);
        var node = child.Nodes[0]; node.Attributes = new() { ["back"] = parent }; child.Nodes[0] = node;
        Assert.Throws<InvalidOperationException>(() => Fingerprint(parent, false));
        Assert.Throws<InvalidOperationException>(() => Fingerprint(parent, true));
        node.Attributes = null; child.Nodes[0] = node;
        Agree(parent);
    }

    [Fact]
    public async Task ContextsShareOnlyImmutableSnapshot_AndInvalidationReleasesOwner()
    {
        var owner = Chain();
        var snapshot = owner.FingerprintStrings!;
        var before = snapshot.ToArray();
        var contexts = new[] { owner.CreateExecution(ExecutionOptions.Memory), owner.CreateExecution(ExecutionOptions.Memory) };
        Assert.All(contexts, context => { Assert.True(context.CacheFingerprintStrings); Assert.Same(snapshot, context.FingerprintStrings); });
        await Task.WhenAll(contexts.Select((context, index) => Task.Run(() =>
        {
            for (int i = 0; i < 10; i++)
            {
                context.Reset();
                Assert.True(context.Execute(Feed(-i, index + i), false), context.LastErrorMessage);
                Assert.Equal(new[] { 0f, (float)(index + i) }, ((Tensor<float>)context.Outputs["y"]).ToArray());
            }
        })));
        Assert.Equal(before, snapshot);
        owner.InvalidatePreparation();
        Assert.Null(owner.FingerprintStrings);
        Assert.All(contexts, context => Assert.Same(snapshot, context.FingerprintStrings));
        Assert.True(contexts[0].Execute(Feed(), false));
        owner.Prepare();
        Assert.NotSame(snapshot, owner.FingerprintStrings);
        Assert.Equal(before, snapshot);
    }
}
