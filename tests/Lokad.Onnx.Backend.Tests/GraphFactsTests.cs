using System.Collections.Generic;
using System.Linq;
using Lokad.Onnx.Optimization;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Checks conservative producer, consumer, output, dtype, dimension and constant facts.
/// </summary>
public class GraphFactsTests
{
    [Fact]
    public void RecurrentOutputsProveFloatOnlyWithMatchingSupportedInputs()
    {
        var graph = new ComputationalGraph{Opset=new(){[""]=17}};
        graph.Inputs["x"] = new DenseTensor<float>(new[]{4,1,2});
        graph.Initializers["w"] = new DenseTensor<float>(new[]{2,8,2});
        graph.Initializers["r"] = new DenseTensor<float>(new[]{2,8,2});
        var node = new Node{Op=OpType.LSTM,OpTypeName="LSTM",Inputs=new[]{"x","w","r"},Outputs=new[]{"y","h","c"},
            Attributes=new(){["hidden_size"]=2,["direction"]="bidirectional"},OpsetVersion=17};
        graph.Nodes.Add(node);
        var facts=GraphFacts.Build(graph);
        foreach(string name in new[]{"y","h","c"}) Assert.Equal(TensorElementType.Float,facts.Dtypes[name]);
        graph.Initializers["r"] = new DenseTensor<double>(new[]{2,8,2});
        Assert.False(GraphFacts.Build(graph).Dtypes.ContainsKey("y"));
        graph.Initializers["r"] = new DenseTensor<float>(new[]{2,8,2});
        node.Domain="custom";graph.Nodes[0]=node;
        Assert.False(GraphFacts.Build(graph).Dtypes.ContainsKey("y"));
    }

    [Fact]
    public void DequantizedDtypeRequiresScaleZeroPointAndAttributesToMatch()
    {
        var graph=new ComputationalGraph{Opset=new(){[""]=17}};
        graph.Initializers["x"]=new DenseTensor<sbyte>(new[]{2,3});
        graph.Initializers["s"]=DenseTensor<float>.OfValues(new[]{.5f});
        graph.Initializers["z"]=DenseTensor<sbyte>.OfValues(new sbyte[]{0});
        var node=new Node{Op=OpType.DequantizeLinear,OpTypeName="DequantizeLinear",Inputs=new[]{"x","s","z"},Outputs=new[]{"y"},Attributes=new(),OpsetVersion=17};
        graph.Nodes.Add(node);
        Assert.Equal(TensorElementType.Float,GraphFacts.Build(graph).Dtypes["y"]);
        graph.Initializers["z"]=DenseTensor<byte>.OfValues(new byte[]{0});
        Assert.False(GraphFacts.Build(graph).Dtypes.ContainsKey("y"));
        graph.Initializers["z"]=DenseTensor<sbyte>.OfValues(new sbyte[]{0});
        node.Attributes["output_dtype"]=(int)TensorElementType.Double;
        Assert.False(GraphFacts.Build(graph).Dtypes.ContainsKey("y"));
    }

    [Theory]
    [InlineData(OpType.Identity)]
    [InlineData(OpType.Clip)]
    [InlineData(OpType.Floor)]
    [InlineData(OpType.GatherElements)]
    [InlineData(OpType.Flatten)]
    [InlineData(OpType.CumSum)]
    [InlineData(OpType.Exp)]
    [InlineData(OpType.Round)]
    [InlineData(OpType.LeakyRelu)]
    [InlineData(OpType.InstanceNormalization)]
    [InlineData(OpType.Pad)]
    public void SupportedValuePreservingOperatorsPropagateDtype(OpType op)
    {
        var graph=new ComputationalGraph{Opset=new(){[""]=17}};
        graph.Inputs["x"]=new DenseTensor<float>(new[]{1,2,3});
        graph.Initializers["i"]=DenseTensor<long>.OfValues(new long[]{0});
        graph.Initializers["s"]=DenseTensor<float>.OfValues(new[]{1f,1f});
        graph.Initializers["b"]=DenseTensor<float>.OfValues(new[]{0f,0f});
        string[] inputs=op switch
        {
            OpType.GatherElements or OpType.CumSum or OpType.Pad=>new[]{"x","i"},
            OpType.InstanceNormalization=>new[]{"x","s","b"},
            _=>new[]{"x"}
        };
        var node=new Node{Op=op,OpTypeName=op.ToString(),Inputs=inputs,Outputs=new[]{"y"},Attributes=new(),OpsetVersion=17};
        graph.Nodes.Add(node);
        Assert.True(CPUExecutionProvider.SupportsNode(node));
        Assert.Equal(TensorElementType.Float,GraphFacts.Build(graph).Dtypes["y"]);
        node.Domain="custom";graph.Nodes[0]=node;
        Assert.False(GraphFacts.Build(graph).Dtypes.ContainsKey("y"));
    }

    [Fact]
    public void FusedMaskedSoftmaxPreservesMatchingInputTypes()
    {
        var graph = new ComputationalGraph { Opset = new() { [""] = 17 } };
        graph.Inputs["x"] = new DenseTensor<float>(new[] { 2, 3 });
        graph.Inputs["mask"] = new DenseTensor<float>(new[] { 3 });
        var node = new Node { Op = OpType.MaskedSoftmax, OpTypeName = "MaskedSoftmax", IsFused = true,
            Inputs = new[] { "x", "mask" }, Outputs = new[] { "y" }, Attributes = new(), OpsetVersion = 17 };
        graph.Nodes.Add(node);
        Assert.Equal(TensorElementType.Float, GraphFacts.Build(graph).Dtypes["y"]);
        graph.Inputs["mask"] = new DenseTensor<double>(new[] { 3 });
        Assert.False(GraphFacts.Build(graph).Dtypes.ContainsKey("y"));
        graph.Inputs["mask"] = new DenseTensor<float>(new[] { 3 });
        node.Domain = "custom"; graph.Nodes[0] = node;
        Assert.False(GraphFacts.Build(graph).Dtypes.ContainsKey("y"));
    }

    [Fact]
    public void ShapeProducesInt64RegardlessOfInputDtype()
    {
        var graph = new ComputationalGraph { Opset = new() { [""] = 17 } };
        graph.Inputs["x"] = new DenseTensor<double>(new[] { 2, 3 });
        graph.Nodes.Add(new Node { Op = OpType.Shape, OpTypeName = "Shape", Inputs = new[] { "x" },
            Outputs = new[] { "y" }, Attributes = new(), OpsetVersion = 17 });
        Assert.Equal(TensorElementType.Int64, GraphFacts.Build(graph).Dtypes["y"]);
    }

    static OnnxModel TinyModel()
    {
        var mp = new OnnxModel { Name = "tiny-facts" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "z", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Initializers.Add(new OnnxTensor { Name = "w", ElementType = TensorElementType.Float, Dims = new[] { 4, 4 }, Data = new float[16] });
        var two = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new int[0], Data = new[] { 2f } });
        mp.Nodes.Add(new OnnxNode { OpType = "Constant", Inputs = new string[0], Outputs = new[] { "c1" }, Attributes = new Dictionary<string, object> { ["value"] = two } });
        mp.Nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { "x", "c1" }, Outputs = new[] { "m" }, Attributes = new Dictionary<string, object>() });
        mp.Nodes.Add(new OnnxNode { OpType = "MatMul", Inputs = new[] { "m", "w" }, Outputs = new[] { "y" }, Attributes = new Dictionary<string, object>() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "m", "x" }, Outputs = new[] { "z" }, Attributes = new Dictionary<string, object>() });
        return mp;
    }

    [Fact]
    public void FactsDescribeTinyGraph()
    {
        var graph = Model.Load(TinyModel())!;
        var facts = GraphFacts.Build(graph);
        Assert.Equal(0, facts.Producer["c1"]);
        Assert.Equal(1, facts.Producer["m"]);
        Assert.Equal(2, facts.Producer["y"]);
        Assert.Equal(3, facts.Producer["z"]);
        Assert.Equal(new[] { 1 }, facts.Consumers["c1"]);
        Assert.Equal(new[] { 1, 3 }, facts.Consumers["x"]);
        Assert.True(facts.IsGraphOutput("y"));
        Assert.True(facts.IsGraphOutput("z"));
        Assert.False(facts.IsGraphOutput("m"));
        Assert.True(facts.IsSingleConsumer("c1"));
        Assert.False(facts.IsSingleConsumer("x"));
        Assert.Equal(TensorElementType.Float, facts.Dtypes["x"]);
        Assert.Equal(TensorElementType.Float, facts.Dtypes["w"]);
        Assert.Equal(TensorElementType.Float, facts.Dtypes["c1"]);
        Assert.Equal(TensorElementType.Float, facts.Dtypes["m"]);
        Assert.Equal(TensorElementType.Float, facts.Dtypes["y"]);
        Assert.Equal(new[] { 2, 4 }, facts.KnownDims["x"]);
        Assert.Equal(new[] { 4, 4 }, facts.KnownDims["w"]);
        Assert.Empty(facts.KnownDims["c1"]);
        Assert.False(facts.KnownDims.ContainsKey("m"));
        Assert.False(facts.Constants.ContainsKey("w"));
        Assert.True(facts.Constants.ContainsKey("c1"));
        Assert.False(facts.Constants.ContainsKey("x"));
    }

    [Fact]
    public void EmptyPassListChangesNothing()
    {
        var graph = Model.Load(TinyModel())!;
        int before = graph.Nodes.Count;
        var report = GraphOptimizer.Run(graph);
        Assert.Empty(report);
        Assert.Equal(before, graph.Nodes.Count);
    }
}
