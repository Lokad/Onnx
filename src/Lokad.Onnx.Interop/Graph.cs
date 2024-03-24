namespace Lokad.Onnx.Interop;

using System;
using System.Collections.Generic;
using System.Linq;

public class Graph : Runtime
{
    public static ComputationalGraph? LoadFromFile(string filepath) => Model.LoadFromFile(filepath);

    public static ITensor[]? GetInputTensorsFromFileArgs(IEnumerable<string> args, bool saveInput = false) => Data.GetInputTensorsFromFileArgs(args, saveInput);

    public static ITensor? GetInputTensorFromFileArg(string arg, bool saveInput = false) => Data.GetInputTensorsFromFileArgs(new[] { arg }, saveInput)?.First();

    public static void SetDebugMode() => Runtime.Initialize("ONNX", "PYTHON", true, true);
}

