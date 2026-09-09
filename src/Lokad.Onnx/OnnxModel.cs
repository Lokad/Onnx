namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// Plain-data ONNX model description with no protobuf dependency. Produced by
/// the Import project from OnnxSharp protos (or built by hand in tests) and
/// consumed by Model.Load to assemble an executable ComputationalGraph.
/// </summary>
public sealed class OnnxModel
{
    public string Name = "";
    public string Domain = "";
    public string DocString = "";
    public string ProducerName = "";
    public string ProducerVersion = "";
    public long IrVersion;
    public Dictionary<string, int> Opset = new Dictionary<string, int>();
    public Dictionary<string, string> MetadataProps = new Dictionary<string, string>();
    public List<OnnxValueInfo> Inputs = new List<OnnxValueInfo>();
    public List<OnnxValueInfo> Outputs = new List<OnnxValueInfo>();
    public List<OnnxTensor> Initializers = new List<OnnxTensor>();
    public List<OnnxNode> Nodes = new List<OnnxNode>();
}

public class OnnxValueInfo
{
    public string Name = "";
    public TensorElementType ElementType;
    /// <summary>
    /// Per-dimension extents. A negative entry marks an anonymous unknown
    /// extent: it accepts any non-negative value without coupling to other
    /// dimensions. Use <see cref="DimParams"/> for shared symbolic extents.
    /// </summary>
    public int[] Dims = Array.Empty<int>();
    /// <summary>
    /// Optional per-dimension symbolic names, aligned with <see cref="Dims"/>.
    /// Null means legacy/unknown (no symbolic info); a null entry means the
    /// dimension is fixed; a non-null entry names a symbolic dimension shared
    /// across the model. Descriptions are treated as immutable after load.
    /// </summary>
    public string?[]? DimParams = null;

    public string Describe() => $"{Name}:{ElementType.ToString().ToLowerInvariant()}:{string.Join("x", Dims.Select(d => d.ToString()))}";
}

public sealed class OnnxTensor : OnnxValueInfo
{
    public Array Data = Array.Empty<float>();
}

public sealed class OnnxNode
{
    public string Name = "";
    public string OpType = "";
    public string Domain = "";
    public string[] Inputs = Array.Empty<string>();
    public string[] Outputs = Array.Empty<string>();
    public Dictionary<string, object> Attributes = new Dictionary<string, object>();
}
