using Google.Protobuf;
using Onnx;

namespace Lokad.Onnx.Backend.Tests;

public class OperatorNameTests
{
    public static IEnumerable<object[]> UnknownNames()
    {
        foreach (string name in new[] { ((int)OpType.Relu).ToString(System.Globalization.CultureInfo.InvariantCulture),
            " Relu ", "Relu, Abs", "999999", "relu", "ExternalAttention" })
        foreach (bool file in new[] { false, true }) yield return new object[] { name, file };
    }

    static OnnxValueInfo Info(string name) => new() { Name = name, ElementType = TensorElementType.Float, Dims = [2] };

    static OnnxModel Description(string name) => new() {
        Name = "operator-name", Opset = { [""] = 18 }, Inputs = { Info("x") }, Outputs = { Info("y") },
        Nodes = { new OnnxNode { Name = "op", OpType = name, Inputs = ["x"], Outputs = ["y"] } } };

    static ComputationalGraph Load(string name, bool file)
    {
        var model = Description(name);
        string domain = name == "ExternalAttention" ? "com.example" : "";
        model.Nodes[0].Domain = domain;
        if (domain.Length != 0) model.Opset[domain] = 1;
        if (!file) return Model.Load(model);
        ValueInfoProto WireInfo(string value) => new() { Name = value, Type = new TypeProto {
            TensorType = new TypeProto.Types.Tensor { ElemType = 1, Shape = new TensorShapeProto {
                Dim = { new TensorShapeProto.Types.Dimension { DimValue = 2 } } } } } };
        var proto = new ModelProto { IrVersion = 10, Graph = new GraphProto { Name = "operator-name",
            Input = { WireInfo("x") }, Output = { WireInfo("y") },
            Node = { new NodeProto { Name = "op", OpType = name, Domain = domain, Input = { "x" }, Output = { "y" } } } },
            OpsetImport = { new OperatorSetIdProto { Domain = "", Version = 18 } } };
        if (domain.Length != 0) proto.OpsetImport.Add(new OperatorSetIdProto { Domain = domain, Version = 1 });
        string path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".onnx");
        try
        {
            File.WriteAllBytes(path, proto.ToByteArray());
            return OnnxImport.LoadOrThrow(path);
        }
        finally { File.Delete(path); }
    }

    static bool Execute(ComputationalGraph graph) => graph.Execute(
        new Dictionary<string, ITensor> { ["x"] = DenseTensor<float>.OfValues(new[] { -1f, 2f }) }, true);

    [Theory]
    [MemberData(nameof(UnknownNames))]
    public void UnknownNamesRemainInspectableButNeverDispatch(string name, bool file)
    {
        var graph = Load(name, file);
        Assert.False(Execute(graph));
        Assert.Equal(OpType.Unknown, graph.Nodes[0].Op);
        Assert.Equal(name, graph.Nodes[0].OpTypeName);
        Assert.False(CPUExecutionProvider.SupportsNode(graph.Nodes[0]));
        Assert.Contains(name, graph.LastErrorMessage);
    }

    [Theory]
    [InlineData("Relu", false)] [InlineData("Relu", true)]
    [InlineData("Abs", false)] [InlineData("Abs", true)]
    [InlineData("Identity", false)] [InlineData("Identity", true)]
    public void LiteralNamesKeepTheirImplementedBehavior(string name, bool file)
    {
        var graph = Load(name, file);
        Assert.True(CPUExecutionProvider.SupportsNode(graph.Nodes[0]));
        Assert.True(Execute(graph), graph.LastErrorMessage);
        float first = name == "Relu" ? 0f : name == "Abs" ? 1f : -1f;
        Assert.Equal(new[] { first, 2f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
    }

    [Theory]
    [InlineData(" Relu ", false)]
    [InlineData("Relu, Abs", false)]
    [InlineData("Relu", true)]
    public void OptimizationDoesNotHideAnUnknownEpilogueName(string name, bool supported)
    {
        var model = Description(name);
        model.Initializers.Add(new OnnxTensor { Name = "bias", ElementType = TensorElementType.Float, Dims = [2], Data = new[] { 0f, 0f } });
        model.Nodes[0].Inputs = ["sum"];
        model.Nodes.Insert(0, new OnnxNode { Name = "add", OpType = "Add", Inputs = ["x", "bias"], Outputs = ["sum"] });
        var graph = Model.Load(model);
        Assert.Equal(supported, Execute(graph));
        if (supported) Assert.Equal(new[] { 0f, 2f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
        else Assert.Contains(name, graph.LastErrorMessage);
    }

    [Theory]
    [InlineData(" Relu ", false)]
    [InlineData("Relu", true)]
    public void NestedBranchesUseLiteralOperatorNames(string name, bool supported)
    {
        OnnxSubgraph Branch(string op) => new() { Name = "branch", Outputs = { Info("y") },
            Nodes = { new OnnxNode { Name = "inner", OpType = op, Inputs = ["x"], Outputs = ["y"] } } };
        var model = Description("If");
        model.Initializers.Add(new OnnxTensor { Name = "condition", ElementType = TensorElementType.Bool, Dims = [], Data = new[] { true } });
        model.Nodes[0].Inputs = ["condition"];
        model.Nodes[0].Attributes = new Dictionary<string, object> {
            ["then_branch"] = Branch(name), ["else_branch"] = Branch("Identity") };
        var graph = Model.Load(model);
        Assert.Equal(supported, Execute(graph));
        if (supported) Assert.Equal(new[] { 0f, 2f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
        else Assert.Contains(name, graph.LastErrorMessage);
    }
}
