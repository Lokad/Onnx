namespace Lokad.Onnx.Backend;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Versioning;

[RequiresPreviewFeatures]
public class ComputationalGraph : Runtime
{
    #region Fields
    public Dictionary<string, ITensor> Inputs = new Dictionary<string, ITensor>();

    public Dictionary<string, ITensor> Outputs = new Dictionary<string, ITensor>();

    public Dictionary<string, ITensor> Initializers = new Dictionary<string, ITensor>();

    public Dictionary<string, ITensor?> IntermediateOutputs = new Dictionary<string, ITensor?>();

    public List<Node> Nodes { get; set; } = new List<Node>();
    
    public WeightedDirectedGraph WeightedDirectedGraph { get; } =  new WeightedDirectedGraph();
    
    public Opset[] Opset = Array.Empty<Opset>();
    
    public Dictionary<string, object> Metadata = new Dictionary<string, object>();

    public Dictionary<string, string> MetadataProps = new Dictionary<string, string>();
    #endregion

    #region Methods
    public int GetOpVersion(string domain) => this.Opset.Single(o => o.Domain == domain).Version;

    public int GetOpVersion() => this.GetOpVersion("");
    
    public ITensor GetTensor(string name) => 
        Inputs.ContainsKey(name) ? Inputs[name] : Initializers.ContainsKey(name) ? Initializers[name]: Outputs[name];
    
    public ITensor[] GetTensors(string[] names) => names.Select(n => GetTensor(n)).ToArray();

    public bool Execute(ITensor[] inputs, ExecutionProvider provider = ExecutionProvider.CPU)
    {
        var userInputs = Inputs.Values.ToList();    
        foreach (var i in Inputs.Keys)
        {
            if (Initializers.ContainsKey(i))
            {
                var ii = Initializers[i];
                var iv = Inputs[i];
                if (ii.Dims.SequenceEqual(iv.Dims) && ii.ElementType == iv.ElementType)
                {
                    Info("Using initializer value {n} for graph input {i}.", ii.TensorNameDesc(), i);
                    Inputs[i] = Initializers[i];
                    userInputs.Remove(iv);
                }
            }
        }
        Info("{uic} user input(s) required for graph: {uig}.", userInputs.Count, userInputs.Select(ui => ui.TensorNameDesc()));   
        return true;
        /*
        foreach(var node in Nodes) 
        {
            if (!node.Inputs.All(i => Inputs.ContainsKey(i) || Initializers.ContainsKey(i)))
        }
        */
    }
    #endregion
}

public class WeightedDirectedGraph : Satsuma.AbstractGraph
{
    public Satsuma.Node AddNode(string id)
    {
        var _id = id.GetHashCode();
        this.AddNode(_id);
        return new Satsuma.Node(_id);
    }
}