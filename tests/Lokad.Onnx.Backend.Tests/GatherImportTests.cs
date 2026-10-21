using Google.Protobuf;
using Onnx;

namespace Lokad.Onnx.Backend.Tests;

public class GatherImportTests
{
    static ValueInfoProto FloatInfo(string name, params long[] dimensions)
    {
        var type = new TypeProto.Types.Tensor { ElemType = 1, Shape = new TensorShapeProto() };
        foreach (long dimension in dimensions)
            type.Shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = dimension });
        return new ValueInfoProto { Name = name, Type = new TypeProto { TensorType = type } };
    }

    [Fact]
    public void ImportedSliceThenScalarGather_MatchesIndependentOrtResult()
    {
        // Identical graph/input independently checked with CPU ORT 1.29, one thread:
        // Slice([0]:[2]) then Gather(index=1, axis=0) returns [4,5,6].
        var model = new ModelProto { IrVersion = 10, Graph = new GraphProto { Name = "slice-scalar-gather" },
            OpsetImport = { new OperatorSetIdProto { Domain = "", Version = 13 } } };
        model.Graph.Input.Add(FloatInfo("data", 2, 3));
        model.Graph.Output.Add(FloatInfo("result", 3));
        model.Graph.Initializer.Add(new TensorProto { Name = "starts", DataType = 7, Dims = { 1 }, Int64Data = { 0 } });
        model.Graph.Initializer.Add(new TensorProto { Name = "ends", DataType = 7, Dims = { 1 }, Int64Data = { 2 } });
        model.Graph.Initializer.Add(new TensorProto { Name = "index", DataType = 7, Int64Data = { 1 } });
        model.Graph.Node.Add(new NodeProto { Name = "slice", OpType = "Slice", Input = { "data", "starts", "ends" }, Output = { "view" } });
        model.Graph.Node.Add(new NodeProto { Name = "gather", OpType = "Gather", Input = { "view", "index" }, Output = { "result" },
            Attribute = { new AttributeProto { Name = "axis", Type = AttributeProto.Types.AttributeType.Int, I = 0 } } });
        var graph = OnnxImport.Load(model.ToByteArray());
        Assert.NotNull(graph);
        var data = new DenseTensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        foreach (var options in new[] { ExecutionOptions.Default, ExecutionOptions.Memory, ExecutionOptions.Scalar })
        {
            Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["data"] = data }, true, ExecutionProvider.CPU, options), graph.LastErrorMessage);
            var result = Assert.IsAssignableFrom<Tensor<float>>(graph.Outputs["result"]);
            Assert.Equal(new[] { 3 }, result.Dimensions.ToArray());
            Assert.Equal(new float[] { 4, 5, 6 }, result.ToArray());
            Assert.Equal(new float[] { 1, 2, 3, 4, 5, 6 }, data.ToArray());
        }
    }
}
