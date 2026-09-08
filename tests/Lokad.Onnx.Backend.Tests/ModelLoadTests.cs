
extern alias OnnxSharp;
using OnnxSharp::Onnx;

namespace Lokad.Onnx.Backend.Tests;

[Collection("SequentialLogSink")]
public class ModelLoadTests
{
    static string MnistModel()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (int i = 0; i < 5; i++) dir = dir!.Parent!;
        return Path.Combine(dir!.FullName, "tests", "Lokad.Onnx.Backend.Tests", "models", "mnist-8.onnx");
    }

    [Fact]
    public void FileAndBufferParse_AgreeOnMnistStructure()
    {
        string path = MnistModel();
        var fromFile = OnnxImport.Parse(path);
        var fromBuffer = OnnxImport.Parse(File.ReadAllBytes(path));
        Assert.Equal(fromBuffer.Inputs.Count, fromFile.Inputs.Count);
        Assert.Equal(fromBuffer.Outputs.Count, fromFile.Outputs.Count);
        Assert.Equal(fromBuffer.Initializers.Count, fromFile.Initializers.Count);
        Assert.Equal(fromBuffer.Nodes.Count, fromFile.Nodes.Count);
        Assert.Equal(
            fromBuffer.Initializers.Select(i => i.Name),
            fromFile.Initializers.Select(i => i.Name));
        Assert.Equal(
            fromBuffer.Nodes.Select(n => n.OpType).OrderBy(o => o),
            fromFile.Nodes.Select(n => n.OpType).OrderBy(o => o));
    }

    [Fact]
    public void ParseMetadata_MatchesParseStructure_WithoutInitializerPayloads()
    {
        string path = MnistModel();
        var full = OnnxImport.Parse(path);
        var meta = OnnxImport.ParseMetadata(path);
        Assert.Equal(full.Inputs.Select(i => i.Describe()), meta.Inputs.Select(i => i.Describe()));
        Assert.Equal(full.Outputs.Select(o => o.Describe()), meta.Outputs.Select(o => o.Describe()));
        Assert.Equal(full.Nodes.Count, meta.Nodes.Count);
        Assert.Equal(full.Initializers.Count, meta.Initializers.Count);
        Assert.Equal(
            full.Initializers.Select(i => i.Describe()),
            meta.Initializers.Select(i => i.Describe()));
        Assert.All(meta.Initializers, i => Assert.Empty(i.Data));
        Assert.All(full.Initializers, i => Assert.True(i.Data.Length > 0));
    }

    [Fact]
    public void ParseMetadata_OpsMatchParse()
    {
        string path = MnistModel();
        var full = OnnxImport.Parse(path);
        var meta = OnnxImport.ParseMetadata(path);
        static System.Collections.Generic.IEnumerable<string> Ops(OnnxModel m) =>
            m.Nodes.Select(n => (n.Domain ?? "") + ":" + (n.OpType ?? "")).Distinct().OrderBy(o => o);
        Assert.Equal(Ops(full), Ops(meta));
    }

    [Fact]
    public void ReadExternalTensorData_ReturnsTypedPayload()
    {
        var dir = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        Directory.CreateDirectory(dir);
        try
        {
            var floats = new float[] { 1f, -2.5f };
            var bytes = new byte[floats.Length * 4];
            Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(Path.Combine(dir, "w.bin"), bytes);
            var proto = new TensorProto { Name = "w", DataType = (int)TensorElementType.Float };
            proto.Dims.Add(2);
            proto.DataLocation = TensorProto.Types.DataLocation.External;
            proto.ExternalData.Add(new StringStringEntryProto { Key = "location", Value = "w.bin" });
            var data = proto.ReadExternalTensorData(dir);
            var typed = Assert.IsType<float[]>(data);
            Assert.Equal(floats, typed);
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    [Fact]
    public void ReadExternalTensorData_SymbolicDims_FallsBackToResolvedBytes()
    {
        var dir = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        Directory.CreateDirectory(dir);
        try
        {
            var floats = new float[] { 1f, 2f };
            var bytes = new byte[floats.Length * 4];
            Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(Path.Combine(dir, "s.bin"), bytes);
            var proto = new TensorProto { Name = "s", DataType = (int)TensorElementType.Float };
            proto.Dims.Add(-1);
            proto.DataLocation = TensorProto.Types.DataLocation.External;
            proto.ExternalData.Add(new StringStringEntryProto { Key = "location", Value = "s.bin" });
            var typed = Assert.IsType<float[]>(proto.ReadExternalTensorData(dir));
            Assert.Equal(floats, typed);
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    [Fact]
    public void ReadExternalTensorData_RejectsNonExternal()
    {
        var proto = new TensorProto { Name = "e", DataType = (int)TensorElementType.Float };
        proto.Dims.Add(1);
        Assert.Throws<System.InvalidOperationException>(() => proto.ReadExternalTensorData(Path.GetTempPath()));
    }

    [Fact]
    public void ModelLoad_TakesOwnershipOfDescription()
    {
        var mp = new OnnxModel
        {
            Name = "own",
            Opset = new System.Collections.Generic.Dictionary<string, int> { { "", 8 } },
            Inputs = new System.Collections.Generic.List<OnnxValueInfo>
            {
                new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new int[] { 2 } },
            },
            Outputs = new System.Collections.Generic.List<OnnxValueInfo>
            {
                new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new int[] { 2 } },
            },
            Initializers = new System.Collections.Generic.List<OnnxTensor>(),
            Nodes = new System.Collections.Generic.List<OnnxNode>
            {
                new OnnxNode { Name = "r", OpType = "Relu", Domain = "", Inputs = new string[] { "x" }, Outputs = new string[] { "y" } },
            },
        };
        var nodeInputs = mp.Nodes[0].Inputs;
        var nodeOutputs = mp.Nodes[0].Outputs;
        var inputs = mp.Inputs;
        var outputs = mp.Outputs;
        var graph = Model.Load(mp);
        Assert.Same(inputs, graph.InputDescs);
        Assert.Same(outputs, graph.OutputDescs);
        Assert.Same(nodeInputs, graph.Nodes[0].Inputs);
        Assert.Same(nodeOutputs, graph.Nodes[0].Outputs);
        var user = new System.Collections.Generic.Dictionary<string, ITensor>
        {
            { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f }) },
        };
        Assert.True(graph.Execute(user, false));
    }
}
