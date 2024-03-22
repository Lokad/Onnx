namespace Lokad.Onnx.Interop;

using System;

public class Graph
{
    public static ComputationalGraph? LoadFromFile(string filepath) => Model.LoadFromFile(filepath);
}

