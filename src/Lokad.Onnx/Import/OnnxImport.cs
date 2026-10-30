namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using global::Onnx;

/// <summary>
/// File and buffer entry points for ONNX models. Parsing (protobuf) and
/// external-data resolution ship in the core package with the generated messages.
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
    /// Parses names, shapes, types and nodes without materializing tensor
    /// payloads, for inspection commands that never read weight values.
    /// Initializers and tensor-valued attributes are OnnxTensor descriptions
    /// with empty payloads, including inside nested graph attributes.
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
        return m.ToModelDto(dir, materializeTensors: false);
    }

    public static OnnxModel Parse(byte[] data)
    {
        using var op = Runtime.Begin("Parsing ONNX model buffer of length {f} bytes", data.Length);
        var m = ModelProto.Parser.ParseFrom(data);
        op.Complete();
        return m.ToModelDto(null, true);
    }

    /// <summary>
    /// Parses a model from caller-owned memory without a <c>ToArray()</c> copy:
    /// the pinned protobuf parser reads <see cref="ReadOnlyMemory{T}.Span"/>
    /// directly and the resulting DTO owns its bytes afterwards.
    /// </summary>
    /// <remarks>
    /// The caller owns the input and keeps it valid and unchanged during the
    /// synchronous call; after return, the parsed model must not depend on it.
    /// </remarks>
    public static OnnxModel Parse(ReadOnlyMemory<byte> data)
    {
        using var op = Runtime.Begin("Parsing ONNX model buffer of length {f} bytes", data.Length);
        var m = ModelProto.Parser.ParseFrom(data.Span);
        op.Complete();
        return m.ToModelDto(null, true);
    }

    /// <summary>Rendered template of the last import failure, without exception text; null when the last load succeeded or none ran. Last write wins under concurrent loads.</summary>
    public static string? LastErrorMessage { get; private set; }

    /// <summary>Original exception behind the last import failure, when one was captured; fatal runtime failures propagate instead and record nothing. Last write wins under concurrent loads.</summary>
    public static Exception? LastErrorCause { get; private set; }

    public static ComputationalGraph? Load(string onnxInputFilePath) =>
        Load(onnxInputFilePath, long.MaxValue);

    /// <summary>Loads a file with a per-graph byte limit on prepared MatMul packed clones.</summary>
    /// <remarks>Zero disables packing. Child graphs receive independent limits of the same size.
    /// Original weights, folded transposes and execution buffers are excluded. Nonfatal failures,
    /// including a negative limit, return null and populate the import diagnostics.</remarks>
    public static ComputationalGraph? Load(string onnxInputFilePath, long maximumPackedWeightBytes) =>
        LoadCore(() => Parse(onnxInputFilePath), "Could not parse {f} as ONNX model file.", "Could not load {f} as ONNX model.", onnxInputFilePath, maximumPackedWeightBytes);

    /// <summary>Loads a local model or throws a call-specific error with the original cause.</summary>
    /// <remarks>Unlike the nullable Load overloads, this method does not read or update
    /// the process-wide LastError properties. Nonfatal failures throw InvalidDataException;
    /// fatal runtime failures propagate unchanged.</remarks>
    public static ComputationalGraph LoadOrThrow(string onnxInputFilePath) => LoadOrThrow(onnxInputFilePath, long.MaxValue);

    /// <summary>Loads a local model with a per-graph packed-weight limit or throws a call-specific error.</summary>
    public static ComputationalGraph LoadOrThrow(string onnxInputFilePath, long maximumPackedWeightBytes) =>
        LoadStages(() => Parse(onnxInputFilePath), "Could not parse {f} as ONNX model file.",
            "Could not load {f} as ONNX model.", onnxInputFilePath, maximumPackedWeightBytes);

    public static ComputationalGraph? Load(byte[] buffer) =>
        LoadCore(() => Parse(buffer), "Could not parse buffer as ONNX model.", "Could not load buffer as ONNX model.", null, long.MaxValue);

    /// <summary>
    /// Loads a model from caller-owned memory without a <c>ToArray()</c> copy,
    /// sharing the buffer overloads contracts: null plus diagnostics on any
    /// nonfatal failure. The caller owns the input during the synchronous call.
    /// </summary>
    public static ComputationalGraph? Load(ReadOnlyMemory<byte> buffer) =>
        LoadCore(() => Parse(buffer), "Could not parse buffer as ONNX model.", "Could not load buffer as ONNX model.", null, long.MaxValue);

    static ComputationalGraph? LoadCore(Func<OnnxModel> parse, string parseTemplate, string loadTemplate, string? path, long maximumPackedWeightBytes)
    {
        LastErrorMessage = null;
        LastErrorCause = null;
        try
        {
            return LoadStages(parse, parseTemplate, loadTemplate, path, maximumPackedWeightBytes);
        }
        catch (InvalidDataException ex)
        {
            LastErrorMessage = ex.Message;
            LastErrorCause = ex.InnerException;
            Runtime.Error(ex.InnerException ?? ex, ex.Message);
            return null;
        }
    }

    static ComputationalGraph LoadStages(Func<OnnxModel> parse, string parseTemplate, string loadTemplate, string? path, long maximumPackedWeightBytes)
    {
        OnnxModel mp;
        try
        {
            ArgumentOutOfRangeException.ThrowIfNegative(maximumPackedWeightBytes);
            mp = parse();
        }
        catch (Exception ex) when (!Runtime.IsFatal(ex))
        {
            throw ImportFailure(ex, parseTemplate, path);
        }
        try
        {
            var g = Model.Load(mp, maximumPackedWeightBytes);
            if (path is not null) g.ModelFile = path;
            return g;
        }
        catch (Exception ex) when (!Runtime.IsFatal(ex))
        {
            throw ImportFailure(ex, loadTemplate, path);
        }
    }

    /// <summary>
    /// Creates an operation-local failure without consulting shared diagnostics.
    /// </summary>
    static InvalidDataException ImportFailure(Exception ex, string messageTemplate, string? path) =>
        new InvalidDataException(Log.Render(messageTemplate, path is null ? Array.Empty<object?>() : new object?[] { path }), ex);

    static OnnxModel ToModelDto(this ModelProto mp, string? baseDirectory, bool materializeTensors)
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
            Initializers = mp.Graph.Initializer.Select(tp => tp.ToTensorDto(baseDirectory, materializeTensors)).ToList(),
            Nodes = mp.Graph.Node.Select(np => np.ToNodeDto(baseDirectory, materializeTensors)).ToList(),
        };
    }
}
