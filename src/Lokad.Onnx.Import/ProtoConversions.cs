namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

public static class ProtoConversions
{
    public static object GetTensorData(this TensorProto tp)
    {
        switch ((TensorElementType)tp.DataType)
        {
            case TensorElementType.Int32:
                Runtime.Debug("tensorproto {tpn} has embedded int32 tensor data.", tp.Name);
                return tp.Int32Data.Count == 0 && tp.RawData.Length > 0 ? MemoryMarshal.Cast<byte, int>(tp.RawData.Span).ToArray() : tp.Int32Data.ToArray();
            case TensorElementType.Int64:
                Runtime.Debug("tensorproto {tpn} has embedded int64 tensor data.", tp.Name);
                return tp.Int64Data.Count == 0 && tp.RawData.Length > 0 ? MemoryMarshal.Cast<byte, long>(tp.RawData.Span).ToArray() : tp.Int64Data.ToArray();
            case TensorElementType.Float:
                Runtime.Debug("tensorproto {tpn} has embedded float tensor data.", tp.Name);
                return tp.FloatData.Count == 0 && tp.RawData.Length > 0 ? MemoryMarshal.Cast<byte, float>(tp.RawData.Span).ToArray() : tp.FloatData.ToArray();
            case TensorElementType.Double:
                Runtime.Debug("tensorproto {tpn} has embedded double tensor data.", tp.Name);
                return tp.DoubleData.Count == 0 && tp.RawData.Length > 0 ? MemoryMarshal.Cast<byte, double>(tp.RawData.Span).ToArray() : tp.DoubleData.ToArray();
            default: throw new NotSupportedException($"Cannot get embedded tensor data of tensor element type {tp.DataType}.");
        }
    }

    /// <summary>
    /// Loads external tensor data addressed by tp from baseDirectory into RawData.
    /// A missing location, a missing file, or an offset range past the end of the file throws.
    /// </summary>
    public static void ResolveExternalData(this TensorProto tp, string baseDirectory)
    {
        if (tp.DataLocation != TensorProto.Types.DataLocation.External) return;
        string? location = null;
        ulong offset = 0;
        ulong length = 0;
        bool hasLength = false;
        foreach (var entry in tp.ExternalData)
        {
            if (entry.Key == "location") location = entry.Value;
            else if (entry.Key == "offset") ulong.TryParse(entry.Value, out offset);
            else if (entry.Key == "length") hasLength = ulong.TryParse(entry.Value, out length);
        }
        if (location is null) throw new InvalidOperationException($"Tensor {tp.Name} references external data without a location.");
        var path = System.IO.Path.Combine(baseDirectory, location);
        using var stream = new System.IO.FileStream(path, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.ReadWrite);
        if (!hasLength) length = (ulong)stream.Length - offset;
        if (offset + length > (ulong)stream.Length) throw new InvalidOperationException($"Tensor {tp.Name} references external data beyond the end of {location}.");
        var buffer = new byte[length];
        stream.Seek((long)offset, System.IO.SeekOrigin.Begin);
        stream.ReadExactly(buffer);
        tp.RawData = Google.Protobuf.ByteString.CopyFrom(buffer);
        tp.DataLocation = TensorProto.Types.DataLocation.Default;
    }

    public static OnnxTensor ToTensorDto(this TensorProto tp)
    {
        return new OnnxTensor
        {
            Name = tp.Name,
            ElementType = (TensorElementType)tp.DataType,
            Dims = tp.Dims.Select(d => Convert.ToInt32(d)).ToArray(),
            Data = (Array)tp.GetTensorData(),
        };
    }

    public static ITensor ToTensor(this TensorProto tp) => Model.ToTensor(tp.ToTensorDto());

    public static OnnxValueInfo ToValueDto(this ValueInfoProto vp)
    {
        if (vp.Type is null || vp.Type.ValueCase != TypeProto.ValueOneofCase.TensorType)
        {
            throw new ArgumentException($"The value info {vp.Name} is not a tensor type.");
        }
        if (vp.Type.TensorType.Shape is null)
        {
            throw new ArgumentException($"The value info {vp.Name} declares a tensor type without shape metadata; shaped graph inputs and outputs are required.");
        }
        return new OnnxValueInfo
        {
            Name = vp.Name,
            ElementType = (TensorElementType)vp.Type.TensorType.ElemType,
            Dims = vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray(),
        };
    }

    public static ITensor ToTensor(this ValueInfoProto vp) => Model.ToShapeTensor(vp.ToValueDto());

    public static string TensorNameDesc(this ValueInfoProto vp) => vp.ToValueDto().Describe();

    public static string TensorNameDesc(this TensorProto tp) => tp.ToTensorDto().Describe();

    public static object Value(this AttributeProto ap)
    {
        switch (ap.Type)
        {
            case AttributeProto.Types.AttributeType.Int: return ap.I;
            case AttributeProto.Types.AttributeType.Ints: return ap.Ints.ToArray();
            case AttributeProto.Types.AttributeType.Float: return ap.F;
            case AttributeProto.Types.AttributeType.Floats: return ap.Floats.ToArray();
            case AttributeProto.Types.AttributeType.Tensor: return ap.T.ToTensor();
            case AttributeProto.Types.AttributeType.String: return ap.S.ToStringUtf8();
            case AttributeProto.Types.AttributeType.Strings: return ap.Strings.Select(s => s.ToStringUtf8()).ToArray();
            default: throw new NotSupportedException($"Cannot convert attribute proto value of type {ap.Type}.");
        }
    }

    public static OnnxNode ToNodeDto(this NodeProto np)
    {
        Runtime.Debug("Converting model node proto {npn} with op type {npot} and inputs {npi} and outputs {npot} and attributes [{npa}] to graph node.", np.Name, np.OpType, np.Input, np.Output, np.Attribute.Select(a => a.Name));
        return new OnnxNode
        {
            Name = np.Name,
            OpType = np.OpType,
            Inputs = np.Input.ToArray(),
            Outputs = np.Output.ToArray(),
            Attributes = np.Attribute.ToDictionary(k => k.Name, v => (object)v.Value()),
        };
    }
}
