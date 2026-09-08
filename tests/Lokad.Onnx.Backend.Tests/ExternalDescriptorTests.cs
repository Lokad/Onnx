extern alias OnnxSharp;

using Google.Protobuf;
using OnnxSharp::Onnx;

namespace Lokad.Onnx.Backend.Tests;

public class ExternalDescriptorTests
{
    static void WithTempDirectory(Action<string> exercise)
    {
        var tempDirectory = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        Directory.CreateDirectory(tempDirectory);
        try
        {
            exercise(tempDirectory);
        }
        finally
        {
            Directory.Delete(tempDirectory, true);
        }
    }

    static TensorProto ExternalProto(string fileName)
    {
        var proto = new TensorProto { Name = "weight", DataType = (int)TensorElementType.Float };
        proto.Dims.Add(2);
        proto.DataLocation = TensorProto.Types.DataLocation.External;
        proto.ExternalData.Add(new StringStringEntryProto { Key = "location", Value = fileName });
        return proto;
    }

    [Fact]
    public void InvalidOffsetSyntax_Throws()
    {
        WithTempDirectory(directory =>
        {
            File.WriteAllBytes(Path.Combine(directory, "w.bin"), new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 });
            foreach (var bad in new string[] { "abc", "12x", "-1", "" })
            {
                var proto = ExternalProto("w.bin");
                proto.ExternalData.Add(new StringStringEntryProto { Key = "offset", Value = bad });
                Assert.Throws<InvalidOperationException>(() => proto.ResolveExternalData(directory));
            }
        });
    }

    [Fact]
    public void InvalidLengthSyntax_Throws()
    {
        WithTempDirectory(directory =>
        {
            File.WriteAllBytes(Path.Combine(directory, "w.bin"), new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 });
            foreach (var bad in new string[] { "xyz", "-8", "8x" })
            {
                var proto = ExternalProto("w.bin");
                proto.ExternalData.Add(new StringStringEntryProto { Key = "length", Value = bad });
                Assert.Throws<InvalidOperationException>(() => proto.ResolveExternalData(directory));
            }
        });
    }

    [Fact]
    public void OffsetPastEnd_Throws()
    {
        WithTempDirectory(directory =>
        {
            File.WriteAllBytes(Path.Combine(directory, "w.bin"), new byte[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            var proto = ExternalProto("w.bin");
            proto.ExternalData.Add(new StringStringEntryProto { Key = "offset", Value = "100" });
            Assert.Throws<InvalidOperationException>(() => proto.ResolveExternalData(directory));
        });
    }

    [Fact]
    public void AbsoluteLocation_Throws()
    {
        WithTempDirectory(directory =>
        {
            var proto = ExternalProto(Path.GetFullPath("evil-abs.bin"));
            Assert.Throws<InvalidOperationException>(() => proto.ResolveExternalData(directory));
        });
    }

    [Fact]
    public void TraversalLocation_Throws()
    {
        WithTempDirectory(directory =>
        {
            var proto = ExternalProto(Path.Combine("..", "evil.bin"));
            Assert.Throws<InvalidOperationException>(() => proto.ResolveExternalData(directory));
        });
    }

    [Fact]
    public void EmptyLocation_Throws()
    {
        WithTempDirectory(directory =>
        {
            var proto = ExternalProto("");
            Assert.Throws<InvalidOperationException>(() => proto.ResolveExternalData(directory));
        });
    }

    [Fact]
    public void ByteCountMismatch_OmittedLength_Throws()
    {
        WithTempDirectory(directory =>
        {
            File.WriteAllBytes(Path.Combine(directory, "short.bin"), new byte[] { 1, 2, 3, 4 });
            var proto = ExternalProto("short.bin");
            Assert.Throws<InvalidOperationException>(() => proto.ResolveExternalData(directory));
        });
    }

    [Fact]
    public void ByteCountMismatch_ExplicitLength_Throws()
    {
        WithTempDirectory(directory =>
        {
            File.WriteAllBytes(Path.Combine(directory, "w.bin"), new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
            var proto = ExternalProto("w.bin");
            proto.ExternalData.Add(new StringStringEntryProto { Key = "offset", Value = "0" });
            proto.ExternalData.Add(new StringStringEntryProto { Key = "length", Value = "4" });
            Assert.Throws<InvalidOperationException>(() => proto.ResolveExternalData(directory));
        });
    }

    [Fact]
    public void SubdirectoryLocation_StaysInside()
    {
        WithTempDirectory(directory =>
        {
            Directory.CreateDirectory(Path.Combine(directory, "sub"));
            var payload = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            File.WriteAllBytes(Path.Combine(directory, "sub", "w.bin"), payload);
            var proto = ExternalProto(Path.Combine("sub", "w.bin"));
            proto.ResolveExternalData(directory);
            Assert.Equal(payload, proto.RawData.ToByteArray());
        });
    }

    [Fact]
    public void TensorAttribute_External_Resolves_With_Directory()
    {
        ExternalDescriptorTests.WithTempDirectory(directory =>
        {
            var payload = new List<byte>();
            payload.AddRange(BitConverter.GetBytes(1.5f));
            payload.AddRange(BitConverter.GetBytes(2.5f));
            File.WriteAllBytes(Path.Combine(directory, "attr.bin"), payload.ToArray());
            var tp = new TensorProto { Name = "w", DataType = (int)TensorElementType.Float };
            tp.Dims.Add(2);
            tp.DataLocation = TensorProto.Types.DataLocation.External;
            tp.ExternalData.Add(new StringStringEntryProto { Key = "location", Value = "attr.bin" });
            var attr = new AttributeProto { Name = "value", Type = AttributeProto.Types.AttributeType.Tensor };
            attr.T = tp;
            var node = new NodeProto { Name = "c", OpType = "Constant" };
            node.Attribute.Add(attr);
            var dto = node.ToNodeDto(directory);
            var tensor = Assert.IsType<DenseTensor<float>>((ITensor)dto.Attributes["value"]);
            Assert.Equal(new float[] { 1.5f, 2.5f }, tensor.ToArray());
        });
    }

    [Fact]
    public void TensorAttribute_External_Without_Directory_Throws()
    {
        var tp = new TensorProto { Name = "w", DataType = (int)TensorElementType.Float };
        tp.Dims.Add(2);
        tp.DataLocation = TensorProto.Types.DataLocation.External;
        tp.ExternalData.Add(new StringStringEntryProto { Key = "location", Value = "attr.bin" });
        var attr = new AttributeProto { Name = "value", Type = AttributeProto.Types.AttributeType.Tensor };
        attr.T = tp;
        var node = new NodeProto { Name = "c", OpType = "Constant" };
        node.Attribute.Add(attr);
        Assert.Throws<InvalidOperationException>(() => node.ToNodeDto(null));
    }

    [Fact]
    public void BufferParse_Rejects_Unresolved_External()
    {
        var init = new TensorProto { Name = "w", DataType = (int)TensorElementType.Float };
        init.Dims.Add(2);
        init.DataLocation = TensorProto.Types.DataLocation.External;
        init.ExternalData.Add(new StringStringEntryProto { Key = "location", Value = "w.bin" });
        var model = new ModelProto();
        model.Graph = new GraphProto();
        model.Graph.Initializer.Add(init);
        var bytes = model.ToByteArray();
        Assert.Throws<InvalidOperationException>(() => OnnxImport.Parse(bytes));
    }
}
