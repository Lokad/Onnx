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
    /// <remarks>
    /// Cost contract: the complete protobuf still parses, including embedded
    /// weights, so metadata inspection is not a cheap header read; only the
    /// typed payload materialization and every sidecar read are skipped.
    /// </remarks>
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

    /// <summary>Rendered template of the last import failure, without exception text; null when the last load succeeded or none ran. Last write wins under concurrent loads.</summary>
    public static string? LastErrorMessage { get; private set; }

    /// <summary>Original exception behind the last import failure, when one was captured; fatal runtime failures propagate instead and record nothing. Last write wins under concurrent loads.</summary>
    public static Exception? LastErrorCause { get; private set; }

    public static ComputationalGraph? Load(string onnxInputFilePath) =>
        LoadCore(() => Parse(onnxInputFilePath), "Could not parse {f} as ONNX model file.", "Could not load {f} as ONNX model.", onnxInputFilePath);

    public static ComputationalGraph? Load(byte[] buffer) =>
        LoadCore(() => Parse(buffer), "Could not parse buffer as ONNX model.", "Could not load buffer as ONNX model.", null);

    static ComputationalGraph? LoadCore(Func<OnnxModel> parse, string parseTemplate, string loadTemplate, string? path)
    {
        LastErrorMessage = null;
        LastErrorCause = null;
        OnnxModel mp;
        try
        {
            mp = parse();
        }
        catch (Exception ex) when (!Runtime.IsFatal(ex))
        {
            return FailImport(ex, parseTemplate, path);
        }
        try
        {
            var g = Model.Load(mp);
            if (path is not null) g.ModelFile = path;
            return g;
        }
        catch (Exception ex) when (!Runtime.IsFatal(ex))
        {
            return FailImport(ex, loadTemplate, path);
        }
    }

    /// <summary>
    /// Records a nonfatal import failure: snapshots the rendered message and
    /// the original cause for readers without a log sink, keeps the log line,
    /// and returns null like every other load rejection.
    /// </summary>
    static ComputationalGraph? FailImport(Exception ex, string messageTemplate, string? path)
    {
        LastErrorCause = ex;
        if (path is null)
        {
            LastErrorMessage = Log.Render(messageTemplate, Array.Empty<object?>());
            Runtime.Error(ex, messageTemplate);
        }
        else
        {
            LastErrorMessage = Log.Render(messageTemplate, new object?[] { path });
            Runtime.Error(ex, messageTemplate, path);
        }
        return null;
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
