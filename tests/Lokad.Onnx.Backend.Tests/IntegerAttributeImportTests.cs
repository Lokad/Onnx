using Google.Protobuf;
using Onnx;

namespace Lokad.Onnx.Backend.Tests;

public class IntegerAttributeImportTests
{
    static ValueInfoProto Value(string name, TensorElementType type, params long[] dims)
    {
        var shape = new TensorShapeProto();
        foreach (long dim in dims) shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = dim });
        return new ValueInfoProto { Name = name, Type = new TypeProto
            { TensorType = new TypeProto.Types.Tensor { ElemType = (int)type, Shape = shape } } };
    }

    static ModelProto ConstantModel(long[] values)
    {
        var model = new ModelProto { IrVersion = 10, Graph = new GraphProto { Name = "integer-list" },
            OpsetImport = { new OperatorSetIdProto { Version = 18 } } };
        model.Graph.Output.Add(Value("y", TensorElementType.Int64, values.Length));
        var attribute = new AttributeProto { Name = "value_ints", Type = AttributeProto.Types.AttributeType.Ints };
        attribute.Ints.Add(values);
        model.Graph.Node.Add(new NodeProto { Name = "constant", OpType = "Constant", Output = { "y" }, Attribute = { attribute } });
        return model;
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(0, 1)]
    [InlineData(0, 2)]
    [InlineData(1, 0)]
    [InlineData(1, 1)]
    [InlineData(1, 2)]
    [InlineData(2, 0)]
    [InlineData(2, 1)]
    [InlineData(2, 2)]
    [InlineData(3, 0)]
    [InlineData(3, 1)]
    [InlineData(3, 2)]
    public void ListConstantsPreserveInt64TypeAndValuesAcrossImportEntryPoints(int payload, int entry)
    {
        // Native ORT produces INT64 even when every value fits in int32.
        long[] values = payload switch
        {
            0 => [1, -2], 1 => [1L << 40, -(1L << 40)],
            2 => [long.MaxValue, long.MinValue], _ => []
        };
        byte[] bytes = ConstantModel(values).ToByteArray();
        string file = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx");
        try
        {
            OnnxModel description;
            ComputationalGraph? graph;
            if (entry == 0)
            {
                File.WriteAllBytes(file, bytes);
                description = OnnxImport.Parse(file);
                graph = OnnxImport.Load(file);
            }
            else if (entry == 1)
            {
                description = OnnxImport.Parse(bytes);
                graph = OnnxImport.Load(bytes);
            }
            else
            {
                var padded = new byte[bytes.Length + 13];
                bytes.CopyTo(padded, 7);
                ReadOnlyMemory<byte> memory = padded.AsMemory(7, bytes.Length);
                description = OnnxImport.Parse(memory);
                graph = OnnxImport.Load(memory);
            }
            Assert.Equal(values, Assert.IsType<long[]>(description.Nodes[0].Attributes["value_ints"]));
            Assert.NotNull(graph);
            Assert.True(graph.Execute(Array.Empty<ITensor>(), true), graph.LastErrorMessage);
            var output = Assert.IsAssignableFrom<Tensor<long>>(graph.Outputs["y"]);
            Assert.Equal(new[] { values.Length }, output.Dimensions.ToArray());
            Assert.Equal(values, output.ToArray());
        }
        finally { if (File.Exists(file)) File.Delete(file); }
    }

    [Fact]
    public void PreparationPreservesHandBuiltConstantPayloadWidth()
    {
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "prepared-list";
        graph.Opset[""] = 18;
        graph.Outputs["y"] = DenseTensor<long>.OfShape(2);
        graph.Nodes.Add(new Node { Op = OpType.Constant, Name = "constant", Inputs = [], Outputs = ["y"],
            Attributes = new() { ["value_ints"] = new long[] { 1, -2 } } });
        graph.RefreshLifetimeAnalysis();
        Assert.IsType<long[]>(graph.Nodes[0].Attributes!["value_ints"]);
        Assert.True(graph.Execute(Array.Empty<ITensor>(), true), graph.LastErrorMessage);
        Assert.Equal(new long[] { 1, -2 }, ((Tensor<long>)graph.Outputs["y"]).ToArray());
    }

    [Theory]
    [InlineData(1L, true)]
    [InlineData(1099511627776L, false)]
    public void GeometryAttributesStillUseCheckedInt32Conversion(long axis, bool supported)
    {
        var attribute = new AttributeProto { Name = "axes", Type = AttributeProto.Types.AttributeType.Ints, Ints = { axis } };
        var parsed = new NodeProto { Name = "squeeze", OpType = "Squeeze", Input = { "x" }, Output = { "y" }, Attribute = { attribute } }.ToNodeDto(null);
        var graph = new ComputationalGraph();
        graph.Inputs["x"] = DenseTensor<float>.OfShape(2, 1);
        var node = new Node { Op = OpType.Squeeze, OpsetVersion = 11, Inputs = ["x"], Outputs = ["y"], Attributes = parsed.Attributes };
        var result = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(supported ? OpStatus.Success : OpStatus.Failure, result.Status);
        if (supported) Assert.Equal(new[] { 2 }, result.Outputs[0].Dims);
        else Assert.IsType<OverflowException>(result.Cause);
    }
}
