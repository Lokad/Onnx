using Lokad.Onnx;

static void Require(bool condition, string message)
{
    if (!condition) throw new InvalidDataException(message);
}

static void Execute(ComputationalGraph graph)
{
    Require(graph.Execute(new ITensor[] { DenseTensor<float>.OfValues([-1, 2]) }, true), graph.LastErrorMessage ?? "Execution failed");
    Require(((Tensor<float>)graph.Outputs["y"]).ToArray().SequenceEqual(new float[] { 0, 2 }), "Unexpected graph output");
}

Execute(Model.Load(CustomReader.Description()));
var reader = new CustomReader();
var file = Model.Load("caller-format.model", reader);
Require(file.ModelFile == "caller-format.model" && reader.Path == file.ModelFile, "File context was lost");
Execute(file);
byte[] bytes = [9, 1, 2, 3, 8];
ReadOnlyMemory<byte> slice = bytes.AsMemory(1, 3);
var memory = Model.Load(slice, reader, 0);
Require(reader.Memory.Equals(slice), "Core copied or changed borrowed reader memory");
Array.Clear(bytes);
Execute(memory);
Require(reader.Calls == 2, "Unexpected reader invocation count");

var core = typeof(Model).Assembly;
Require(!core.GetReferencedAssemblies().Any(a => a.Name == "Google.Protobuf" || a.Name == "Lokad.Onnx.Protobuf"), "Core references a parser assembly");
Require(!core.GetExportedTypes().Any(t => t.Namespace == "Onnx"), "Core exports generated messages");
Require(!AppDomain.CurrentDomain.GetAssemblies().Any(a => a.GetName().Name == "Google.Protobuf" || a.GetName().Name == "Lokad.Onnx.Protobuf"), "Consumer loaded a parser assembly");
Console.WriteLine("SMOKE-CORE-OK DTO, custom file reader, borrowed memory reader; no Protobuf");
Console.WriteLine(System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription);

// A synthetic format reader proves dependency injection without a parser library.
sealed class CustomReader : IOnnxModelReader
{
    public int Calls;
    public string? Path;
    public ReadOnlyMemory<byte> Memory;

    public OnnxModel Read(string path) { Calls++; Path = path; return Description(); }
    public OnnxModel Read(ReadOnlyMemory<byte> data) { Calls++; Memory = data; return Description(); }

    public static OnnxModel Description() => new()
    {
        Name = "package-consumer", Opset = { [""] = 13 },
        Inputs = { new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = [2] } },
        Outputs = { new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = [2] } },
        Nodes = { new OnnxNode { Name = "relu", OpType = "Relu", Inputs = ["x"], Outputs = ["y"] } }
    };
}
