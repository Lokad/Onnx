namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

public static class ProtoConversions
{
    public static object GetTensorData(this TensorProto tp)
    {
        var elementType = (TensorElementType)tp.DataType;
        int size = TensorBase.ElementByteSize(elementType);
        if (tp.RawData.Length > 0 && size > 0 && tp.RawData.Length % size != 0)
            throw new InvalidOperationException($"Tensor {tp.Name} holds {tp.RawData.Length} raw bytes, not a multiple of {size} for {elementType}.");
        var data = (Array)DecodeTensorData(tp, elementType);
        RequireElementCount(tp, data);
        return data;
    }

    static void RequireElementCount(TensorProto tp, Array data)
    {
        long expected;
        try
        {
            checked
            {
                expected = 1;
                foreach (var d in tp.Dims)
                {
                    if (d < 0) return; // Underivable (symbolic) extents skip the check, matching
                        // the external-data byte-count policy for such shapes.
                    expected *= d;
                }
            }
        }
        catch (OverflowException)
        {
            throw new InvalidOperationException($"Tensor {tp.Name} declares a shape whose element count overflows.");
        }
        if (data.Length != expected)
            throw new InvalidOperationException($"Tensor {tp.Name} declares {expected} elements but its payload holds {data.Length}.");
    }

    static object DecodeTensorData(TensorProto tp, TensorElementType elementType)
    {
        switch (elementType)
        {
            case TensorElementType.Bool:
                Runtime.Debug("tensorproto {tpn} has embedded bool tensor data.", tp.Name);
                if (tp.RawData.Length > 0) return MemoryMarshal.Cast<byte, bool>(tp.RawData.Span).ToArray();
                return tp.Int32Data.Select(v => v != 0).ToArray();
            case TensorElementType.Int8:
                Runtime.Debug("tensorproto {tpn} has embedded int8 tensor data.", tp.Name);
                if (tp.RawData.Length > 0) return MemoryMarshal.Cast<byte, sbyte>(tp.RawData.Span).ToArray();
                if (tp.Int64Data.Count > 0) return tp.Int64Data.Select(v => checked((sbyte)v)).ToArray();
                return tp.Int32Data.Select(v => checked((sbyte)v)).ToArray();
            case TensorElementType.UInt8:
                Runtime.Debug("tensorproto {tpn} has embedded uint8 tensor data.", tp.Name);
                if (tp.RawData.Length > 0) return tp.RawData.ToByteArray();
                if (tp.Int64Data.Count > 0) return tp.Int64Data.Select(v => checked((byte)v)).ToArray();
                return tp.Int32Data.Select(v => checked((byte)v)).ToArray();
            case TensorElementType.Int16:
                Runtime.Debug("tensorproto {tpn} has embedded int16 tensor data.", tp.Name);
                if (tp.RawData.Length > 0) return MemoryMarshal.Cast<byte, short>(tp.RawData.Span).ToArray();
                if (tp.Int64Data.Count > 0) return tp.Int64Data.Select(v => checked((short)v)).ToArray();
                return tp.Int32Data.Select(v => checked((short)v)).ToArray();
            case TensorElementType.UInt16:
                Runtime.Debug("tensorproto {tpn} has embedded uint16 tensor data.", tp.Name);
                if (tp.RawData.Length > 0) return MemoryMarshal.Cast<byte, ushort>(tp.RawData.Span).ToArray();
                if (tp.Int64Data.Count > 0) return tp.Int64Data.Select(v => checked((ushort)v)).ToArray();
                return tp.Int32Data.Select(v => checked((ushort)v)).ToArray();
            case TensorElementType.Int32:
                Runtime.Debug("tensorproto {tpn} has embedded int32 tensor data.", tp.Name);
                if (tp.RawData.Length > 0) return MemoryMarshal.Cast<byte, int>(tp.RawData.Span).ToArray();
                if (tp.Int64Data.Count > 0) return tp.Int64Data.Select(v => checked((int)v)).ToArray();
                return tp.Int32Data.ToArray();
            case TensorElementType.UInt32:
                Runtime.Debug("tensorproto {tpn} has embedded uint32 tensor data.", tp.Name);
                if (tp.RawData.Length > 0) return MemoryMarshal.Cast<byte, uint>(tp.RawData.Span).ToArray();
                if (tp.Uint64Data.Count > 0) return tp.Uint64Data.Select(v => checked((uint)v)).ToArray();
                if (tp.Int64Data.Count > 0) return tp.Int64Data.Select(v => checked((uint)v)).ToArray();
                return tp.Int32Data.Select(v => checked((uint)v)).ToArray();
            case TensorElementType.Int64:
                Runtime.Debug("tensorproto {tpn} has embedded int64 tensor data.", tp.Name);
                if (tp.RawData.Length > 0) return MemoryMarshal.Cast<byte, long>(tp.RawData.Span).ToArray();
                if (tp.Int64Data.Count > 0) return tp.Int64Data.ToArray();
                return tp.Int32Data.Select(v => (long)v).ToArray();
            case TensorElementType.UInt64:
                Runtime.Debug("tensorproto {tpn} has embedded uint64 tensor data.", tp.Name);
                if (tp.RawData.Length > 0) return MemoryMarshal.Cast<byte, ulong>(tp.RawData.Span).ToArray();
                if (tp.Uint64Data.Count > 0) return tp.Uint64Data.ToArray();
                if (tp.Int64Data.Count > 0) return tp.Int64Data.Select(v => checked((ulong)v)).ToArray();
                return tp.Int32Data.Select(v => checked((ulong)v)).ToArray();
            case TensorElementType.Float:
                Runtime.Debug("tensorproto {tpn} has embedded float tensor data.", tp.Name);
                return tp.FloatData.Count == 0 && tp.RawData.Length > 0 ? MemoryMarshal.Cast<byte, float>(tp.RawData.Span).ToArray() : tp.FloatData.ToArray();
            case TensorElementType.Double:
                Runtime.Debug("tensorproto {tpn} has embedded double tensor data.", tp.Name);
                return tp.DoubleData.Count == 0 && tp.RawData.Length > 0 ? MemoryMarshal.Cast<byte, double>(tp.RawData.Span).ToArray() : tp.DoubleData.ToArray();
            case TensorElementType.Float16:
                Runtime.Debug("tensorproto {tpn} has embedded float16 tensor data.", tp.Name);
                if (tp.RawData.Length > 0) return MemoryMarshal.Cast<ushort, Half>(MemoryMarshal.Cast<byte, ushort>(tp.RawData.Span)).ToArray();
                return tp.Int32Data.Select(v => BitConverter.UInt16BitsToHalf(checked((ushort)v))).ToArray();
            case TensorElementType.BFloat16:
                Runtime.Debug("tensorproto {tpn} has embedded bfloat16 tensor data.", tp.Name);
                if (tp.RawData.Length > 0) return MemoryMarshal.Cast<ushort, Lokad.Onnx.BFloat16>(MemoryMarshal.Cast<byte, ushort>(tp.RawData.Span)).ToArray();
                return tp.Int32Data.Select(v => new Lokad.Onnx.BFloat16(checked((ushort)v))).ToArray();
            default: throw new NotSupportedException($"Tensor {tp.Name} has unsupported element type {elementType} and cannot be imported.");
        }
    }

    /// <summary>
    /// Loads external tensor data addressed by tp from baseDirectory into RawData.
    /// A missing location, a missing file, malformed or out-of-range offset/length
    /// entries, a location escaping the model directory, or a byte count that does
    /// not match the tensor descriptor throws. Missing files surface the underlying
    /// FileNotFoundException.
    /// </summary>
    readonly record struct ExternalSegment(string Path, ulong Offset, ulong Length);

    static ExternalSegment GetExternalSegment(TensorProto tp, string baseDirectory)
    {
        /// <summary>
        /// Expected external byte count from dims and element type, or null when the
        /// shape has symbolic (non-positive) dims or the type has no fixed size.
        /// Returns null rather than throwing: callers skip the check in that case.
        /// </summary>
        ulong? ExpectedByteCount(TensorProto tp)
        {
            int size = TensorBase.ElementByteSize((TensorElementType)tp.DataType);
            if (size < 0) return null;
            try
            {
                checked
                {
                    ulong count = 1;
                    foreach (var d in tp.Dims)
                    {
                        if (d <= 0) return null;
                        count *= (ulong)d;
                    }
                    return count * (ulong)size;
                }
            }
            catch (OverflowException)
            {
                return null;
            }
        }

        string? location = null;
        ulong offset = 0;
        ulong length = 0;
        bool hasLength = false;
        foreach (var entry in tp.ExternalData)
        {
            if (entry.Key == "location") location = entry.Value;
            else if (entry.Key == "offset")
            {
                if (!ulong.TryParse(entry.Value, out offset)) throw new InvalidOperationException($"Tensor {tp.Name} has an invalid external-data offset {entry.Value}.");
            }
            else if (entry.Key == "length")
            {
                if (!ulong.TryParse(entry.Value, out length)) throw new InvalidOperationException($"Tensor {tp.Name} has an invalid external-data length {entry.Value}.");
                hasLength = true;
            }
        }
        if (string.IsNullOrEmpty(location)) throw new InvalidOperationException($"Tensor {tp.Name} references external data without a location.");
        // Supported location policy: paths relative to the model directory that
        // stay inside it. Absolute locations would discard the base directory
        // in Path.Combine, and parent traversal escapes it.
        if (System.IO.Path.IsPathRooted(location)) throw new InvalidOperationException($"Tensor {tp.Name} references external data with an absolute location {location}.");
        var baseFull = System.IO.Path.GetFullPath(baseDirectory);
        var path = System.IO.Path.GetFullPath(System.IO.Path.Combine(baseFull, location));
        if (path != baseFull && !path.StartsWith(baseFull + System.IO.Path.DirectorySeparatorChar, StringComparison.Ordinal)) throw new InvalidOperationException($"Tensor {tp.Name} references external data outside the model directory: {location}.");
        using var stream = new System.IO.FileStream(path, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.ReadWrite);
        var fileLength = (ulong)stream.Length;
        if (offset > fileLength) throw new InvalidOperationException($"Tensor {tp.Name} references external data at offset {offset} past the end of {location} ({fileLength} bytes).");
        // Subtraction only, so no wraparound: offset <= fileLength here.
        var remaining = fileLength - offset;
        var effectiveLength = hasLength ? length : remaining;
        if (effectiveLength > remaining) throw new InvalidOperationException($"Tensor {tp.Name} references external data beyond the end of {location}.");
        var expected = ExpectedByteCount(tp);
        if (expected.HasValue && effectiveLength != expected.Value) throw new InvalidOperationException($"Tensor {tp.Name} references {effectiveLength} external bytes but its shape needs {expected.Value}.");
        if (effectiveLength > int.MaxValue) throw new InvalidOperationException($"Tensor {tp.Name} references {effectiveLength} external bytes, which cannot be buffered.");
        return new ExternalSegment(path, offset, effectiveLength);
    }

    public static void ResolveExternalData(this TensorProto tp, string baseDirectory)
    {
        if (tp.DataLocation != TensorProto.Types.DataLocation.External) return;
        var segment = GetExternalSegment(tp, baseDirectory);
        using var stream = new System.IO.FileStream(segment.Path, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.ReadWrite);
        if (segment.Length > int.MaxValue) throw new InvalidOperationException($"Tensor {tp.Name} references {segment.Length} external bytes, which cannot be buffered.");
        var buffer = new byte[(int)segment.Length];
        stream.Seek((long)segment.Offset, System.IO.SeekOrigin.Begin);
        stream.ReadExactly(buffer);
        tp.RawData = Google.Protobuf.ByteString.CopyFrom(buffer);
        tp.DataLocation = TensorProto.Types.DataLocation.Default;
    }

    /// <summary>
    /// Reads an external tensor segment directly into its final typed array.
    /// Unlike resolving into the parsed bytes and converting afterwards, this
    /// performs a single copy from the sidecar file into caller-owned storage.
    /// All C17 location and range validation still applies. Tensors whose
    /// element count cannot be derived (symbolic dims, unsupported types) fall
    /// back to the resolving path. Throws when the tensor is not external.
    /// </summary>
    public static Array ReadExternalTensorData(this TensorProto tp, string baseDirectory)
    {
        if (tp.DataLocation != TensorProto.Types.DataLocation.External)
            throw new InvalidOperationException($"Tensor {tp.Name} does not reference external data.");
        var segment = GetExternalSegment(tp, baseDirectory);
        var elementType = (TensorElementType)tp.DataType;
        int size = TensorBase.ElementByteSize(elementType);
        ulong count = 0;
        bool direct = size > 0;
        if (direct)
        {
            try
            {
                checked
                {
                    count = 1;
                    foreach (var d in tp.Dims)
                    {
                        if (d <= 0) { direct = false; break; }
                        count *= (ulong)d;
                    }
                }
            }
            catch (OverflowException)
            {
                direct = false;
            }
        }
        if (!direct)
        {
            tp.ResolveExternalData(baseDirectory);
            return (Array)tp.GetTensorData();
        }
        if (count > int.MaxValue / (ulong)size || count * (ulong)size != segment.Length)
            throw new InvalidOperationException($"Tensor {tp.Name} references {segment.Length} external bytes but its shape needs {count * (ulong)size}.");
        int length = (int)count;
        Array data = TensorBase.CreateElementArray(elementType, length);
        using var stream = new System.IO.FileStream(segment.Path, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.ReadWrite, 1 << 16, System.IO.FileOptions.SequentialScan);
        System.Span<byte> bytes = data switch
        {
            bool[] b => MemoryMarshal.AsBytes(b.AsSpan()),
            sbyte[] sb => MemoryMarshal.Cast<sbyte, byte>(sb.AsSpan()),
            byte[] ub => ub.AsSpan(),
            short[] s => MemoryMarshal.Cast<short, byte>(s.AsSpan()),
            ushort[] us => MemoryMarshal.Cast<ushort, byte>(us.AsSpan()),
            Half[] h => MemoryMarshal.Cast<Half, byte>(h.AsSpan()),
            Lokad.Onnx.BFloat16[] bh => MemoryMarshal.Cast<Lokad.Onnx.BFloat16, byte>(bh.AsSpan()),
            int[] i => MemoryMarshal.Cast<int, byte>(i.AsSpan()),
            uint[] ui => MemoryMarshal.Cast<uint, byte>(ui.AsSpan()),
            float[] f => MemoryMarshal.Cast<float, byte>(f.AsSpan()),
            long[] l => MemoryMarshal.Cast<long, byte>(l.AsSpan()),
            ulong[] ul => MemoryMarshal.Cast<ulong, byte>(ul.AsSpan()),
            double[] d => MemoryMarshal.Cast<double, byte>(d.AsSpan()),
            _ => throw new NotSupportedException($"Tensor {tp.Name} has unsupported element type {elementType} and cannot be imported."),
        };
        stream.Seek((long)segment.Offset, System.IO.SeekOrigin.Begin);
        stream.ReadExactly(bytes);
        return data;
    }

    /// <summary>
    /// Converts a tensor proto to its plain-data description. External tensors
    /// are read directly into their final typed array when a base directory is
    /// given; without one they throw as before. When materialize is false the
    /// description carries names, types and shapes but no payload, for
    /// metadata-only inspection without the weight cost.
    /// </summary>
    public static OnnxTensor ToTensorDto(this TensorProto tp, string? baseDirectory, bool materialize)
    {
        Array data;
        if (tp.DataLocation == TensorProto.Types.DataLocation.External)
        {
            if (baseDirectory is null)
                throw new InvalidOperationException($"Tensor {tp.Name} references external data, which was not resolved. Load the model from its file so external data can be read.");
            data = materialize ? tp.ReadExternalTensorData(baseDirectory) : Array.Empty<float>();
        }
        else
        {
            data = materialize ? (Array)tp.GetTensorData() : Array.Empty<float>();
        }
        return new OnnxTensor
        {
            Name = tp.Name,
            ElementType = (TensorElementType)tp.DataType,
            Dims = tp.Dims.Select(d => Convert.ToInt32(d)).ToArray(),
            Data = data,
        };
    }

    public static ITensor ToTensor(this TensorProto tp) => Model.ToTensor(tp.ToTensorDto(null, true));

    /// <summary>Converts a tensor proto to an executable tensor, reading external
    /// segments directly into final storage when a base directory is given.</summary>
    public static ITensor ToTensor(this TensorProto tp, string? baseDirectory)
    {
        if (tp.DataLocation == TensorProto.Types.DataLocation.External && baseDirectory is not null)
            return Model.ToTensor(new OnnxTensor
            {
                Name = tp.Name,
                ElementType = (TensorElementType)tp.DataType,
                Dims = tp.Dims.Select(d => Convert.ToInt32(d)).ToArray(),
                Data = tp.ReadExternalTensorData(baseDirectory),
            });
        return Model.ToTensor(tp.ToTensorDto(baseDirectory, true));
    }

    public static OnnxValueInfo ToValueDto(this ValueInfoProto vp)
    {
        if (vp.Type is not null && vp.Type.ValueCase == TypeProto.ValueOneofCase.SequenceType)
        {
            return new OnnxValueInfo
            {
                Name = vp.Name,
                ElementType = TensorElementType.Sequence,
                Dims = Array.Empty<int>(),
                DimParams = Array.Empty<string?>(),
            };
        }
        if (vp.Type is null || vp.Type.ValueCase != TypeProto.ValueOneofCase.TensorType)
        {
            throw new ArgumentException($"The value info {vp.Name} is not a tensor or sequence type.");
        }
        if (vp.Type.TensorType.Shape is null)
        {
            throw new ArgumentException($"The value info {vp.Name} declares a tensor type without shape metadata; shaped graph inputs and outputs are required.");
        }
        return new OnnxValueInfo
        {
            Name = vp.Name,
            ElementType = (TensorElementType)vp.Type.TensorType.ElemType,
            Dims = vp.Type.TensorType.Shape.Dim.Select(d => d.ValueCase switch
            {
                TensorShapeProto.Types.Dimension.ValueOneofCase.DimValue => Convert.ToInt32(d.DimValue),
                TensorShapeProto.Types.Dimension.ValueOneofCase.DimParam => 0,
                // Anonymous unknown dimension: neither value nor name; distinct from fixed zero.
                _ => -1,
            }).ToArray(),
            DimParams = vp.Type.TensorType.Shape.Dim.Select(d => string.IsNullOrEmpty(d.DimParam) ? null : d.DimParam).ToArray(),
        };
    }

    public static string TensorNameDesc(this ValueInfoProto vp) => vp.ToValueDto().Describe();

    public static string TensorNameDesc(this TensorProto tp) => $"{tp.Name}:{((TensorElementType)tp.DataType).ToString().ToLowerInvariant()}:{string.Join("x", tp.Dims)}";

    public static object Value(this AttributeProto ap, string? baseDirectory)
    {
        switch (ap.Type)
        {
            case AttributeProto.Types.AttributeType.Int: return ap.I;
            case AttributeProto.Types.AttributeType.Ints: return ap.Ints.Select(v => checked((int)v)).ToArray();
            case AttributeProto.Types.AttributeType.Float: return ap.F;
            case AttributeProto.Types.AttributeType.Floats: return ap.Floats.ToArray();
            case AttributeProto.Types.AttributeType.Tensor: return ap.T.ToTensor(baseDirectory);
            case AttributeProto.Types.AttributeType.String: return ap.S.ToStringUtf8();
            case AttributeProto.Types.AttributeType.Strings: return ap.Strings.Select(s => s.ToStringUtf8()).ToArray();
            default: throw new NotSupportedException($"Cannot convert attribute proto value of type {ap.Type}.");
        }
    }

    public static OnnxNode ToNodeDto(this NodeProto np, string? baseDirectory)
    {
        Runtime.Debug("Converting model node proto {npn} with op type {npot} and inputs {npi} and outputs {npot} and attributes [{npa}] to graph node.", np.Name, np.OpType, np.Input, np.Output, np.Attribute.Select(a => a.Name));
        // Tensor-valued attributes read external data directly into final
        // storage like top-level initializers do; without a directory the
        // conversion below rejects unresolved external tensors with a clear
        // error. Attribute tensors are small constants, so they always
        // materialize, even on metadata-only paths.
        return new OnnxNode
        {
            Name = np.Name,
            OpType = np.OpType,
            Domain = np.Domain ?? "",
            Inputs = np.Input.ToArray(),
            Outputs = np.Output.ToArray(),
            Attributes = np.Attribute.ToDictionary(k => k.Name, v => (object)v.Value(baseDirectory)),
        };
    }
}
