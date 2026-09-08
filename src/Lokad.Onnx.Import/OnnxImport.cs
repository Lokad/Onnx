namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

/// <summary>
/// File and buffer entry points for ONNX models. Parsing (protobuf) and
/// external-data resolution live here so the core library stays free of the
/// OnnxSharp dependency; referenced by the CLI, tests and runners only.
/// </summary>
public static class OnnxImport
{
    public static OnnxModel Parse(string onnxInputFilePath)
    {
        using var op = Runtime.Begin("Parsing ONNX model file {f}", onnxInputFilePath);
        var dir = Path.GetDirectoryName(Path.GetFullPath(onnxInputFilePath));
        using var stream = new FileStream(onnxInputFilePath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.SequentialScan);
        var m = ModelProto.Parser.ParseFrom(stream);
        op.Complete();
        return m.ToModelDto(dir, true);
    }

    /// <summary>
    /// Parses names, shapes, types and nodes without materializing initializer
    /// payloads, for inspection commands that never read weight values.
    /// Initializer descriptions report element types and dimensions with empty
    /// payloads; attribute tensors still materialize as small constants.
    /// </summary>
    public static OnnxModel ParseMetadata(string onnxInputFilePath)
    {
        using var op = Runtime.Begin("Parsing ONNX model metadata {f}", onnxInputFilePath);
        var dir = Path.GetDirectoryName(Path.GetFullPath(onnxInputFilePath));
        using var stream = new FileStream(onnxInputFilePath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, FileOptions.SequentialScan);
        var m = ModelProto.Parser.ParseFrom(stream);
        op.Complete();
        return m.ToModelDto(dir, materializeInitializers: false);
    }

    public static OnnxModel Parse(byte[] data)
    {
        using var op = Runtime.Begin("Parsing ONNX model buffer of length {f} bytes", data.Length);
        var m = ModelProto.Parser.ParseFrom(data);
        op.Complete();
        return m.ToModelDto(null, true);
    }

    public static ComputationalGraph? Load(string onnxInputFilePath)
    {
        OnnxModel mp;
        try
        {
            mp = Parse(onnxInputFilePath);
        }
        catch (Exception ex)
        {
            Runtime.Error(ex, "Could not parse {f} as ONNX model file.", onnxInputFilePath);
            return null;
        }
        try
        {
            var g = Model.Load(mp);
            g.ModelFile = onnxInputFilePath;
            return g;
        }
        catch (Exception ex)
        {
            Runtime.Error(ex, "Could not load {f} as ONNX model.", onnxInputFilePath);
            return null;
        }
    }

    public static ComputationalGraph? Load(byte[] buffer)
    {
        OnnxModel mp;
        try
        {
            mp = Parse(buffer);
        }
        catch (Exception ex)
        {
            Runtime.Error(ex, "Could not parse buffer as ONNX model.");
            return null;
        }
        try
        {
            return Model.Load(mp);
        }
        catch (Exception ex)
        {
            Runtime.Error(ex, "Could not load buffer as ONNX model.");
            return null;
        }
    }

    static OnnxModel ToModelDto(this ModelProto mp, string? baseDirectory, bool materializeInitializers)
    {
        return new OnnxModel
        {
            Name = mp.Graph.Name,
            Domain = mp.Domain,
            DocString = mp.DocString,
            ProducerName = mp.ProducerName,
            ProducerVersion = mp.ProducerVersion,
            IrVersion = mp.IrVersion,
            Opset = mp.OpsetImport.ToDictionary(o => o.Domain, o => Convert.ToInt32(o.Version)),
            MetadataProps = mp.MetadataProps.ToDictionary(p => p.Key, p => p.Value),
            Inputs = mp.Graph.Input.Select(vp => vp.ToValueDto()).ToList(),
            Outputs = mp.Graph.Output.Select(vp => vp.ToValueDto()).ToList(),
            Initializers = mp.Graph.Initializer.Select(tp => tp.ToTensorDto(baseDirectory, materializeInitializers)).ToList(),
            Nodes = mp.Graph.Node.Select(np => np.ToNodeDto(baseDirectory)).ToList(),
        };
    }
}
