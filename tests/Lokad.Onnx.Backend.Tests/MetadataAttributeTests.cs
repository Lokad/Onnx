namespace Lokad.Onnx.Backend.Tests;

using Google.Protobuf;
using global::Onnx;

public class MetadataAttributeTests
{
    static ValueInfoProto Output(string name) => new() { Name = name, Type = new TypeProto
        { TensorType = new TypeProto.Types.Tensor { ElemType = 1,
            Shape = new TensorShapeProto { Dim = { new TensorShapeProto.Types.Dimension { DimValue = 1 } } } } } };

    static GraphProto Graph(int depth, TensorProto tensor)
    {
        string output = "value-" + depth;
        var graph = new GraphProto { Name = "metadata-" + depth, Output = { Output(output) } };
        if (depth == 0)
            graph.Node.Add(new NodeProto { Name = "constant", OpType = "Constant", Output = { output },
                Attribute = { new AttributeProto { Name = "value", Type = AttributeProto.Types.AttributeType.Tensor, T = tensor } } });
        else
        {
            string condition = "condition-" + depth;
            graph.Initializer.Add(new TensorProto { Name = condition, DataType = 9, Int32Data = { 1 } });
            graph.Node.Add(new NodeProto { Name = "choose", OpType = "If", Input = { condition }, Output = { output }, Attribute = {
                new AttributeProto { Name = "then_branch", Type = AttributeProto.Types.AttributeType.Graph, G = Graph(depth - 1, tensor.Clone()) },
                new AttributeProto { Name = "else_branch", Type = AttributeProto.Types.AttributeType.Graph, G = Graph(depth - 1, tensor.Clone()) } } });
        }
        return graph;
    }

    static void WithModel(int depth, bool external, Action<string, string, ModelProto> exercise)
    {
        string directory = Path.Combine(Path.GetTempPath(), "lonnx-metadata-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            var tensor = new TensorProto { Name = "weight", DataType = 1, Dims = { 1 } };
            if (external)
            {
                tensor.DataLocation = TensorProto.Types.DataLocation.External;
                tensor.ExternalData.Add(new StringStringEntryProto { Key = "location", Value = "weights.bin" });
            }
            else tensor.FloatData.Add(.5f);
            var model = new ModelProto { IrVersion = 8, Graph = Graph(depth, tensor),
                OpsetImport = { new OperatorSetIdProto { Domain = "", Version = 14 } } };
            string file = Path.Combine(directory, "model.onnx");
            File.WriteAllBytes(file, model.ToByteArray());
            exercise(file, directory, model);
        }
        finally
        {
            foreach (string file in Directory.EnumerateFiles(directory)) File.Delete(file);
            Directory.Delete(directory);
        }
    }

    static IEnumerable<object> TensorAttributes(IEnumerable<OnnxNode> nodes)
    {
        foreach (var node in nodes)
            foreach (var (name, value) in node.Attributes)
                if (value is OnnxSubgraph branch)
                {
                    foreach (var attribute in TensorAttributes(branch.Nodes)) yield return attribute;
                }
                else if (name == "value") yield return value;
    }

    [Theory]
    [InlineData(0, false)] [InlineData(0, true)]
    [InlineData(2, false)] [InlineData(2, true)]
    public void MetadataAttributesDescribeShapesWithoutPayloads(int depth, bool external)
    {
        WithModel(depth, external, (file, _, _) =>
        {
            var attributes = TensorAttributes(OnnxImport.ParseMetadata(file).Nodes).ToArray();
            Assert.Equal(1 << depth, attributes.Length);
            Assert.All(attributes, value =>
            {
                var tensor = Assert.IsType<OnnxTensor>(value);
                Assert.Equal("weight", tensor.Name);
                Assert.Equal(TensorElementType.Float, tensor.ElementType);
                Assert.Equal(new[] { 1 }, tensor.Dims);
                Assert.Empty(tensor.Data);
            });
        });
    }

    [Theory]
    [InlineData(0, false)] [InlineData(0, true)]
    [InlineData(2, false)] [InlineData(2, true)]
    public void FullImportStillMaterializesAttributeValues(int depth, bool external)
    {
        WithModel(depth, external, (file, directory, _) =>
        {
            if (external)
            {
                Assert.Throws<FileNotFoundException>(() => OnnxImport.Parse(file));
                Assert.Throws<InvalidOperationException>(() => OnnxImport.Parse(new ReadOnlyMemory<byte>(File.ReadAllBytes(file))));
                File.WriteAllBytes(Path.Combine(directory, "weights.bin"), BitConverter.GetBytes(.5f));
            }
            var full = OnnxImport.Parse(file);
            Assert.All(TensorAttributes(full.Nodes), value => Assert.Equal(new[] { .5f }, Assert.IsAssignableFrom<Tensor<float>>(value).ToArray()));
            var graph = Model.Load(full);
            Assert.True(graph.Execute(Array.Empty<ITensor>(), true), graph.LastErrorMessage);
            Assert.Equal(new[] { .5f }, ((Tensor<float>)graph.Outputs["value-" + depth]).ToArray());
        });
    }

    [Theory]
    [InlineData(0)] [InlineData(2)]
    public void MetadataDoesNotResolveInvalidSidecarLocations(int depth)
    {
        WithModel(depth, true, (file, _, model) =>
        {
            void Change(GraphProto graph)
            {
                foreach (var attribute in graph.Node.SelectMany(n => n.Attribute))
                    if (attribute.T is { } tensor) tensor.ExternalData[0].Value = "../outside.bin";
                    else if (attribute.G is { } child) Change(child);
            }
            Change(model.Graph);
            File.WriteAllBytes(file, model.ToByteArray());
            Assert.All(TensorAttributes(OnnxImport.ParseMetadata(file).Nodes), value => Assert.Empty(Assert.IsType<OnnxTensor>(value).Data));
            Assert.Throws<InvalidOperationException>(() => OnnxImport.Parse(file));
        });
    }

    [Fact]
    public void MetadataStillRejectsExcessiveGeometry()
    {
        WithModel(0, false, (file, _, model) =>
        {
            model.Graph.Node[0].Attribute[0].T.Dims[0] = 1L << 49;
            File.WriteAllBytes(file, model.ToByteArray());
            Assert.Throws<ArgumentException>(() => OnnxImport.ParseMetadata(file));
        });
    }
}
