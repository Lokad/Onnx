namespace Lokad.Onnx.Protobuf;

/// <summary>Reference reader whose Protobuf runtime is owned by this adapter.</summary>
public sealed class ProtobufModelReader : IOnnxModelReader
{
    /// <inheritdoc />
    public OnnxModel Read(string path) => OnnxImport.Parse(path);

    /// <inheritdoc />
    public OnnxModel Read(ReadOnlyMemory<byte> data) => OnnxImport.Parse(data);
}
