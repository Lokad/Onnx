using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Q01 provenance: only standard-domain Constant nodes seed folding.
/// Overridable inputs, replaced or mutated initializers, graph outputs,
/// custom-domain producers and failing probes must all preserve plain
/// (unoptimized) behavior bit for bit.
/// </summary>
public class ConstFoldProvenanceTests
{
    static OnnxModel OverrideModel()
    {
        var mp = new OnnxModel { Name = "tiny-override" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "z", ElementType = TensorElementType.Float, Dims = new int[0] });
        mp.Initializers.Add(new OnnxTensor { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2 }, Data = new float[] { 1f, 2f } });
        var idx = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Int64, Dims = new int[0], Data = new long[] { 0 } });
        mp.Nodes.Add(new OnnxNode { OpType = "Constant", Domain = "", Inputs = new string[0], Outputs = new[] { "idx" }, Attributes = new Dictionary<string, object> { ["value"] = idx } });
        mp.Nodes.Add(new OnnxNode { OpType = "Gather", Domain = "", Inputs = new[] { "x", "idx" }, Outputs = new[] { "z" }, Attributes = new Dictionary<string, object>() });
        return mp;
    }

    static float GatherScalar(ComputationalGraph graph, Dictionary<string, ITensor> feed, bool useInitializers)
    {
        Assert.True(graph.Execute(feed, useInitializers), graph.LastErrorMessage);
        return ((Tensor<float>)graph.Outputs["z"]).ToArray()[0];
    }

    static Dictionary<string, ITensor> Feed98() =>
        new Dictionary<string, ITensor> { ["x"] = DenseTensor<float>.OfValues(new float[] { 9f, 8f }) };

    [Fact]
    public void OverridingInputBeatsInitializer()
    {
        var opt = Model.Load(OverrideModel())!;
        var plain = Model.Load(OverrideModel(), runOptimizer: false)!;
        Assert.Equal(9f, GatherScalar(opt, Feed98(), false));
        Assert.Equal(9f, GatherScalar(plain, Feed98(), false));
        var empty = new Dictionary<string, ITensor>();
        Assert.Equal(1f, GatherScalar(opt, empty, true));
        Assert.Equal(1f, GatherScalar(plain, empty, true));
    }

    [Fact]
    public void ReplacedInitializerPlusInvalidationAgrees()
    {
        var opt = Model.Load(OverrideModel())!;
        var plain = Model.Load(OverrideModel(), runOptimizer: false)!;
        var swap = DenseTensor<float>.OfValues(new float[] { 5f, 6f });
        opt.Initializers["x"] = swap;
        plain.Initializers["x"] = swap;
        opt.InvalidatePreparation();
        plain.InvalidatePreparation();
        var empty = new Dictionary<string, ITensor>();
        Assert.Equal(5f, GatherScalar(opt, empty, true));
        Assert.Equal(5f, GatherScalar(plain, empty, true));
    }

    [Fact]
    public void MutatedInitializerPlusInvalidationAgrees()
    {
        var opt = Model.Load(OverrideModel())!;
        var plain = Model.Load(OverrideModel(), runOptimizer: false)!;
        ((DenseTensor<float>)opt.Initializers["x"]).Buffer.Span[0] = 3f;
        ((DenseTensor<float>)plain.Initializers["x"]).Buffer.Span[0] = 3f;
        opt.InvalidatePreparation();
        plain.InvalidatePreparation();
        var empty = new Dictionary<string, ITensor>();
        Assert.Equal(3f, GatherScalar(opt, empty, true));
        Assert.Equal(3f, GatherScalar(plain, empty, true));
    }

    [Fact]
    public void GraphOutputShapeChainKeepsItsNode()
    {
        var mp = new OnnxModel { Name = "tiny-output-chain" };
        mp.Opset[""] = 14;
        mp.Outputs.Add(new OnnxValueInfo { Name = "z", ElementType = TensorElementType.Int64, Dims = new[] { 1 } });
        var c = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new[] { 3 }, Data = new float[] { 10f, 20f, 30f } });
        mp.Nodes.Add(new OnnxNode { OpType = "Constant", Domain = "", Inputs = new string[0], Outputs = new[] { "c" }, Attributes = new Dictionary<string, object> { ["value"] = c } });
        mp.Nodes.Add(new OnnxNode { OpType = "Shape", Domain = "", Inputs = new[] { "c" }, Outputs = new[] { "z" }, Attributes = new Dictionary<string, object>() });
        var graph = Model.Load(mp)!;
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Shape && n.Outputs.Length == 1 && n.Outputs[0] == "z");
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.Constant && n.Outputs.Length == 1 && n.Outputs[0] == "z");
        Assert.True(graph.Execute(new Dictionary<string, ITensor>(), true), graph.LastErrorMessage);
        Assert.Equal(new long[] { 3 }, ((Tensor<long>)graph.Outputs["z"]).ToArray());
    }

    [Fact]
    public void CustomDomainProducerPreservesRejection()
    {
        var mp = new OnnxModel { Name = "tiny-custom" };
        mp.Opset[""] = 14;
        mp.Outputs.Add(new OnnxValueInfo { Name = "z", ElementType = TensorElementType.Int64, Dims = new[] { 1 } });
        var c = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new[] { 2, 3 }, Data = new float[] { 1f, 2f, 3f, 4f, 5f, 6f } });
        mp.Nodes.Add(new OnnxNode { OpType = "Constant", Domain = "custom.review", Inputs = new string[0], Outputs = new[] { "c" }, Attributes = new Dictionary<string, object> { ["value"] = c } });
        mp.Nodes.Add(new OnnxNode { OpType = "Shape", Domain = "", Inputs = new[] { "c" }, Outputs = new[] { "z" }, Attributes = new Dictionary<string, object>() });
        var plain = Model.Load(mp, runOptimizer: false)!;
        var opt = Model.Load(mp)!;
        Assert.False(plain.Execute(new Dictionary<string, ITensor>(), true));
        Assert.False(opt.Execute(new Dictionary<string, ITensor>(), true));
        Assert.Contains(opt.Nodes, n => n.Outputs.Length == 1 && n.Outputs[0] == "c");
    }

    [Fact]
    public void FailingProbePreservesRejection()
    {
        var mp = new OnnxModel { Name = "tiny-bad-gather" };
        mp.Opset[""] = 14;
        mp.Outputs.Add(new OnnxValueInfo { Name = "z", ElementType = TensorElementType.Float, Dims = new int[0] });
        var data = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new[] { 2 }, Data = new float[] { 1f, 2f } });
        var idx = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Int64, Dims = new int[0], Data = new long[] { 5 } });
        mp.Nodes.Add(new OnnxNode { OpType = "Constant", Domain = "", Inputs = new string[0], Outputs = new[] { "d" }, Attributes = new Dictionary<string, object> { ["value"] = data } });
        mp.Nodes.Add(new OnnxNode { OpType = "Constant", Domain = "", Inputs = new string[0], Outputs = new[] { "i" }, Attributes = new Dictionary<string, object> { ["value"] = idx } });
        mp.Nodes.Add(new OnnxNode { OpType = "Gather", Domain = "", Inputs = new[] { "d", "i" }, Outputs = new[] { "z" }, Attributes = new Dictionary<string, object>() });
        var plain = Model.Load(mp, runOptimizer: false)!;
        var opt = Model.Load(mp)!;
        Assert.Contains(opt.Nodes, n => n.Op == OpType.Gather);
        Assert.False(plain.Execute(new Dictionary<string, ITensor>(), true));
        Assert.False(opt.Execute(new Dictionary<string, ITensor>(), true));
    }
}
