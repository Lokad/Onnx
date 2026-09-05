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
        var op = Runtime.Begin("Parsing ONNX model file {f}", onnxInputFilePath);
        var buffer = File.ReadAllBytes(onnxInputFilePath);
        var m = ModelProto.Parser.ParseFrom(buffer);
        op.Complete();
        var dir = Path.GetDirectoryName(Path.GetFullPath(onnxInputFilePath));
        foreach (var init in m.Graph.Initializer)
        {
            init.ResolveExternalData(dir!);
        }
        return m.ToModelDto();
    }

    public static OnnxModel Parse(byte[] data)
    {
        var op = Runtime.Begin("Parsing ONNX model buffer of length {f} bytes", data.Length);
        var m = ModelProto.Parser.ParseFrom(data);
        op.Complete();
        return m.ToModelDto();
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
        var g = Model.Load(mp);
        g.ModelFile = onnxInputFilePath;
        return g;
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
        return Model.Load(mp);
    }

    static OnnxModel ToModelDto(this ModelProto mp)
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
            Initializers = mp.Graph.Initializer.Select(tp => tp.ToTensorDto()).ToList(),
            Nodes = mp.Graph.Node.Select(np => np.ToNodeDto()).ToList(),
        };
    }
}
