
using Google.Protobuf;
using Onnx;
using Lokad.Onnx.Tests.Support;

namespace Lokad.Onnx.Backend.Tests;

[Collection("SequentialLogSink")]
public class ModelLoadTests
{
    static string MnistModel() => TestSupport.CommittedModel("mnist-8.onnx");

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
    public void ReadOnlyMemoryParse_AgreesWithBufferParse()
    {
        string path = MnistModel();
        byte[] bytes = File.ReadAllBytes(path);
        var fromBuffer = OnnxImport.Parse(bytes);
        var fromMemory = OnnxImport.Parse(new ReadOnlyMemory<byte>(bytes));
        Assert.Equal(fromBuffer.Nodes.Count, fromMemory.Nodes.Count);
        Assert.Equal(fromBuffer.Initializers.Count, fromMemory.Initializers.Count);
        Assert.Equal(
            fromBuffer.Nodes.Select(n => n.OpType).OrderBy(o => o),
            fromMemory.Nodes.Select(n => n.OpType).OrderBy(o => o));
        Assert.Equal(
            fromBuffer.Initializers.Select(i => i.Name),
            fromMemory.Initializers.Select(i => i.Name));
    }

    [Fact]
    public void ReadOnlyMemorySlice_ParsesWithoutArrayCopy()
    {
        // Offset view: proves the overload reads the span, not the backing array.
        string path = MnistModel();
        byte[] bytes = File.ReadAllBytes(path);
        var padded = new byte[bytes.Length + 64];
        Array.Copy(bytes, 0, padded, 37, bytes.Length);
        var slice = new ReadOnlyMemory<byte>(padded, 37, bytes.Length);
        var fromSlice = OnnxImport.Parse(slice);
        var fromBuffer = OnnxImport.Parse(bytes);
        Assert.Equal(fromBuffer.Nodes.Count, fromSlice.Nodes.Count);
        Assert.Equal(fromBuffer.Initializers.Count, fromSlice.Initializers.Count);
    }

    [Fact]
    public void ReadOnlyMemoryLoad_ExecutesMnist()
    {
        string path = MnistModel();
        byte[] bytes = File.ReadAllBytes(path);
        var g = OnnxImport.Load(new ReadOnlyMemory<byte>(bytes));
        Assert.NotNull(g);
        Assert.Single(g!.Outputs);
    }

    sealed class NativeByteOwner : System.Buffers.MemoryManager<byte>
    {
        private unsafe byte* _ptr;
        private int _len;
        private bool _disposed;
        public unsafe NativeByteOwner(byte[] source)
        {
            _len = source.Length;
            _ptr = (byte*)System.Runtime.InteropServices.Marshal.AllocHGlobal(_len).ToPointer();
            new System.Span<byte>(_ptr, _len).Clear();
            source.CopyTo(new System.Span<byte>(_ptr, _len));
        }
        public override System.Span<byte> GetSpan()
        {
            unsafe { return new System.Span<byte>(_ptr, _len); }
        }
        public override unsafe System.Buffers.MemoryHandle Pin(int elementIndex = 0)
        {
            return new System.Buffers.MemoryHandle(_ptr + elementIndex);
        }
        public override void Unpin() { }
        protected override void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                unsafe { System.Runtime.InteropServices.Marshal.FreeHGlobal(new System.IntPtr(_ptr)); }
                _disposed = true;
            }
        }
    }

    [Fact]
    public void ReadOnlyMemoryNonArrayBacked_ParsesWithoutArray()
    {
        // Unmanaged-backed memory: MemoryMarshal.TryGetArray fails on it, so any
        // array assumption in the path throws instead of silently copying.
        string path = MnistModel();
        byte[] bytes = File.ReadAllBytes(path);
        using var owner = new NativeByteOwner(bytes);
        System.ReadOnlyMemory<byte> rom = owner.Memory;
        var fromNative = OnnxImport.Parse(rom);
        var fromBuffer = OnnxImport.Parse(bytes);
        Assert.Equal(fromBuffer.Nodes.Count, fromNative.Nodes.Count);
        Assert.Equal(fromBuffer.Initializers.Count, fromNative.Initializers.Count);
        var g = OnnxImport.Load(rom);
        Assert.NotNull(g);
        Assert.Single(g!.Outputs);
    }

    [Fact]
    public void ReadOnlyMemoryReuseAfterReturn_LeavesDtoIntact()
    {
        string path = MnistModel();
        byte[] bytes = File.ReadAllBytes(path);
        var padded = new byte[bytes.Length + 64];
        Array.Copy(bytes, 0, padded, 37, bytes.Length);
        var slice = new ReadOnlyMemory<byte>(padded, 37, bytes.Length);
        var dto = OnnxImport.Parse(slice);
        int nodes = dto.Nodes.Count;
        string firstInit = dto.Initializers[0].Name;
        int firstLen = dto.Initializers[0].Data.Length;
        Array.Fill<byte>(padded, 0xFF);
        Assert.Equal(nodes, dto.Nodes.Count);
        Assert.Equal(firstInit, dto.Initializers[0].Name);
        Assert.Equal(firstLen, dto.Initializers[0].Data.Length);
    }

    [Fact]
    public void ReadOnlyMemoryInvalid_MatchesBufferContracts()
    {
        var garbage = new byte[] { 1, 2, 3, 4 };
        Exception? bufferEx = null, memoryEx = null;
        try { OnnxImport.Parse(garbage); } catch (Exception ex) { bufferEx = ex; }
        try { OnnxImport.Parse(new ReadOnlyMemory<byte>(garbage)); } catch (Exception ex) { memoryEx = ex; }
        Assert.NotNull(bufferEx);
        Assert.NotNull(memoryEx);
        Assert.Equal(bufferEx!.GetType(), memoryEx!.GetType());
        Assert.Null(OnnxImport.Load(garbage));
        Assert.Null(OnnxImport.Load(new ReadOnlyMemory<byte>(garbage)));
        Assert.NotNull(OnnxImport.LastErrorMessage);
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
    public void ParseMetadata_SkipsSidecarReads()
    {
        var dir = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        Directory.CreateDirectory(dir);
        try
        {
            var model = new ModelProto { Graph = new GraphProto() };
            var init = new TensorProto { Name = "w", DataType = (int)TensorElementType.Float };
            init.Dims.Add(2);
            init.DataLocation = TensorProto.Types.DataLocation.External;
            init.ExternalData.Add(new StringStringEntryProto { Key = "location", Value = "w.bin" });
            model.Graph.Initializer.Add(init);
            var path = Path.Combine(dir, "m.onnx");
            File.WriteAllBytes(path, model.ToByteArray());
            var meta = OnnxImport.ParseMetadata(path);
            var described = Assert.Single(meta.Initializers);
            Assert.Equal("w", described.Name);
            Assert.Empty(described.Data);
            Assert.Throws<System.IO.FileNotFoundException>(() => OnnxImport.Parse(path));
        }
        finally
        {
            Directory.Delete(dir, true);
        }
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
