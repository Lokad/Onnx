namespace Lokad.Onnx;

using System.Collections.Generic;

using Satsuma;

using Lokad.Onnx.Tensors;
  
public class WeightedDirectedGraph: AbstractGraph
{

}
public class ComputationalGraph
{
    public Dictionary<string, TensorBase> Inputs { get; set; } = new Dictionary<string, TensorBase>();
    public Dictionary<string, TensorBase> Outputs { get; set; } = new Dictionary<string, TensorBase>();

    public List<Node> Nodes { get; set; } = new List<Node>();
}
