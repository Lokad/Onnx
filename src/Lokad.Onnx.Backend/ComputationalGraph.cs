﻿namespace Lokad.Onnx.Backend;

using System.Collections.Generic;
using System.Linq;

public class WeightedDirectedGraph: Satsuma.AbstractGraph
{
    public Satsuma.Node AddNode(string id)
    {
        var _id = id.GetHashCode();
        this.AddNode(_id);
        return new Satsuma.Node(_id);
    }
}
public class ComputationalGraph : Runtime
{
    #region Fields
    public Dictionary<string, ITensor> Inputs = new Dictionary<string, ITensor>();

    public Dictionary<string, ITensor> Outputs = new Dictionary<string, ITensor>();

    public List<Node> Nodes { get; set; } = new List<Node>();
    
    public WeightedDirectedGraph WeightedDirectedGraph { get; } =  new WeightedDirectedGraph();
    
    public Dictionary<string, object> Props = new Dictionary<string, object>();

    public Dictionary<string, string> MetadataProps = new Dictionary<string, string>();
    #endregion

    #region Methods
    public Node AddNode(string name)
    {
        Node node = new Node()
        {
            Name = name,
        };
        return node;
    }

   
    public ITensor GetTensor(string name) => Inputs.ContainsKey(name)? Inputs[name] : Outputs[name];
    
    public ITensor[] GetTensors(string[] names) => names.Select(n => GetTensor(n)).ToArray();

    #endregion
}
