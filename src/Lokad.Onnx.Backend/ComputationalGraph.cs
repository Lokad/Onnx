namespace Lokad.Onnx;

using System.Collections.Generic;


public class WeightedDirectedGraph: Satsuma.AbstractGraph
{
    public Satsuma.Node AddNode(string id)
    {
        var _id = id.GetHashCode();
        this.AddNode(_id);
        return new Satsuma.Node(_id);
    }
}
public class ComputationalGraph
{
    #region Properties
    public Dictionary<string, TensorBase> Inputs { get; set; } = new Dictionary<string, TensorBase>();
    
    public Dictionary<string, TensorBase> Outputs { get; set; } = new Dictionary<string, TensorBase>();

    public List<Node> Nodes { get; set; } = new List<Node>();
    
    public WeightedDirectedGraph WeightedDirectedGraph { get; } =  new WeightedDirectedGraph();
    #endregion

    public Node AddNode(string name)
    {
        Node node = new Node()
        {
            Name = name,
        };
        return node;
    }
}
