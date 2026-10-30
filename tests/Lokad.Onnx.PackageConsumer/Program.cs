using Google.Protobuf;
using Lokad.Onnx;
using global::Onnx;

static void Require(bool condition, string message)
{
    if (!condition) throw new InvalidDataException(message);
}

static void Execute(ComputationalGraph graph)
{
    Require(graph.Execute(new ITensor[] { DenseTensor<float>.OfValues([-2, 1]) }, true), graph.LastErrorMessage ?? "Execution failed");
    Require(((Tensor<float>)graph.Outputs["y"]).ToArray().SequenceEqual(new float[] { 0, 3 }), "Unexpected graph output");
}

static ValueInfoProto Value(string name) => new()
{
    Name = name,
    Type = new TypeProto { TensorType = new TypeProto.Types.Tensor
    {
        ElemType = (int)TensorProto.Types.DataType.Float,
        Shape = new TensorShapeProto { Dim = { new TensorShapeProto.Types.Dimension { DimValue = 2 } } }
    } }
};

var bias = new TensorProto { Name = "bias", DataType = (int)TensorProto.Types.DataType.Float, Dims = { 2 }, FloatData = { 1, 2 } };
var proto = new ModelProto
{
    IrVersion = 9, OpsetImport = { new OperatorSetIdProto { Domain = "", Version = 13 } },
    Graph = new GraphProto
    {
        Name = "package-consumer", Input = { Value("x") }, Output = { Value("y") }, Initializer = { bias },
        Node =
        {
            new NodeProto { Name = "add", OpType = "Add", Input = { "x", "bias" }, Output = { "sum" } },
            new NodeProto { Name = "relu", OpType = "Relu", Input = { "sum" }, Output = { "y" } }
        }
    }
};
Require(((Tensor<float>)bias.ToTensor()).ToArray().SequenceEqual(new float[] { 1, 2 }), "Public protobuf tensor conversion failed");
Execute(Model.Load(new OnnxModel
{
    Name = "plain-description", Opset = { [""] = 13 },
    Inputs = { new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = [2] } },
    Outputs = { new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = [2] } },
    Initializers = { bias.ToTensorDto(null, true) },
    Nodes = { proto.Graph.Node[0].ToNodeDto(null), proto.Graph.Node[1].ToNodeDto(null) }
}));

byte[] serialized = proto.ToByteArray();
Execute(OnnxImport.Load(serialized) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage));
byte[] borrowed = new byte[serialized.Length + 4];
serialized.CopyTo(borrowed.AsSpan(2));
var memory = OnnxImport.Load(borrowed.AsMemory(2, serialized.Length)) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
Array.Clear(borrowed);
Execute(memory);
var file = Path.Combine(AppContext.BaseDirectory, "consumer-" + Guid.NewGuid().ToString("N") + ".onnx");
try
{
    File.WriteAllBytes(file, serialized);
    var graph = OnnxImport.Load(file) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
    Require(graph.ModelFile == file, "File context was lost");
    Execute(graph);
}
finally { File.Delete(file); }

var core = typeof(Model).Assembly;
Require(core.GetReferencedAssemblies().Any(a => a.Name == "Google.Protobuf"), "Core is missing its protobuf runtime reference");
Require(typeof(ModelProto).Assembly == core && typeof(OnnxImport).Assembly == core && typeof(ProtoConversions).Assembly == core, "Public importer/messages are outside the core");
Require(core.GetType("Lokad.Onnx.IOnnxModelReader") is null, "Reader injection API remains");
Require(!AppDomain.CurrentDomain.GetAssemblies().Any(a => a.GetName().Name == "Lokad.Onnx.Protobuf"), "Consumer loaded a separate parser assembly");
Console.WriteLine("SMOKE-CORE-OK generated messages, conversions, DTO, direct file/byte/memory import; Google.Protobuf only");
Console.WriteLine(System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription);
