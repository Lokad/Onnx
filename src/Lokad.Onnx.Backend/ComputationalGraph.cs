namespace Lokad.Onnx.Backend;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Versioning;

public class WeightedDirectedGraph: Satsuma.AbstractGraph
{
    public Satsuma.Node AddNode(string id)
    {
        var _id = id.GetHashCode();
        this.AddNode(_id);
        return new Satsuma.Node(_id);
    }
}
[RequiresPreviewFeatures]
public class ComputationalGraph : Runtime
{
    #region Fields
    public Dictionary<string, ITensor> Inputs = new Dictionary<string, ITensor>();

    public Dictionary<string, ITensor> Outputs = new Dictionary<string, ITensor>();

    public List<Node> Nodes { get; set; } = new List<Node>();
    
    public WeightedDirectedGraph WeightedDirectedGraph { get; } =  new WeightedDirectedGraph();
    
    public Opset[] Opset = Array.Empty<Opset>();
    
    public Dictionary<string, object> Metadata = new Dictionary<string, object>();

    public Dictionary<string, string> MetadataProps = new Dictionary<string, string>();


    #endregion

    #region Methods
    public int GetOpVersion(string domain) => this.Opset.Single(o => o.Domain == domain).Version;

    public int GetOpVersion() => this.GetOpVersion("");
    
    public ITensor GetTensor(string name) => Inputs.ContainsKey(name)? Inputs[name] : Outputs[name];
    
    public ITensor[] GetTensors(string[] names) => names.Select(n => GetTensor(n)).ToArray();
    #endregion
}
