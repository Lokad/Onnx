namespace Lokad.Onnx.Backend;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Versioning;

[RequiresPreviewFeatures]
public class ComputationalGraph : Runtime
{
    #region Fields
    public string ModelFile = "";
    
    public Dictionary<string, ITensor> Inputs = new Dictionary<string, ITensor>();

    public Dictionary<string, ITensor> Outputs = new Dictionary<string, ITensor>();

    public Dictionary<string, ITensor> Initializers = new Dictionary<string, ITensor>();

    public Dictionary<string, ITensor?> IntermediateOutputs = new Dictionary<string, ITensor?>();

    public List<Node> Nodes { get; set; } = new List<Node>();

    public WeightedDirectedGraph WeightedDirectedGraph { get; } = new WeightedDirectedGraph();

    public Opset[] Opset = Array.Empty<Opset>();

    public Dictionary<string, object> Metadata = new Dictionary<string, object>();

    public Dictionary<string, string> MetadataProps = new Dictionary<string, string>();
    #endregion

    #region Methods
    public int GetOpVersion(string domain) => this.Opset.Single(o => o.Domain == domain).Version;

    public int GetOpVersion() => this.GetOpVersion("");

    public ITensor GetInputTensor(string name) =>
        Inputs.ContainsKey(name) ? Inputs[name] : Initializers.ContainsKey(name) ? Initializers[name] : 
            IntermediateOutputs[name] ?? throw new InvalidOperationException($"The intermediate output tensor {name} has not been assigned a value.");

    public ITensor[] GetInputTensors(string[] names) => names.Select(n => GetInputTensor(n)).ToArray();

    public bool ResolveInputs(ITensor[] userInputs)
    {
        var op = Begin("Resolving {c} graph inputs for execution", Inputs.Count);
        var requiredInputs = new Dictionary<string, ITensor>(Inputs);
        foreach (var i in Inputs.Keys)
        {
            if (Initializers.ContainsKey(i))
            {
                var ii = Initializers[i];
                var iv = Inputs[i];
                if (ii.Dims.SequenceEqual(iv.Dims) && ii.ElementType == iv.ElementType)
                {
                    Info("Using initializer value {n} for graph input {i}.", ii.TensorNameDesc(), Inputs[i].TensorNameDesc());
                    Inputs[i] = Initializers[i];
                    requiredInputs.Remove(i);
                }
                else
                {
                    Error("Cannot use initializer value {n} for graph input {i}. Tensor shape or type does not match.", ii.TensorNameDesc(), Inputs[i].TensorNameDesc());
                    op.Abandon();
                    return false;
                }
            }
        }
        Info("{uic} user input(s) required for graph execution: {uig}.", requiredInputs.Count, requiredInputs.Select(ui => ui.Value.TensorNameDesc()));
        if (userInputs.Length != requiredInputs.Count)
        {
            Error("{uic} user input(s) required for graph execution:{i} but only {c} specified.", requiredInputs.Count, userInputs.Select(ui => ui.TensorNameDesc()), userInputs.Length);
            op.Abandon();
            return false;
        }
        for(int i = 0; i < requiredInputs.Keys.Count; i++)
        {
            if (!userInputs[i].Dims.SequenceEqual(requiredInputs.ElementAt(i).Value.Dims) || !(userInputs[i].ElementType == requiredInputs.ElementAt(i).Value.ElementType))
            {
                Error("Cannot use user input {ui} for required input {ri}. Tensor shape or type does not match.", userInputs[i].TensorNameDesc(), requiredInputs.ElementAt(i).Value.TensorNameDesc());
                op.Abandon();
                return false;
            }
            else
            {
                Info("Using user input {n} for graph input {i}.", userInputs[i].TensorNameDesc(), requiredInputs.ElementAt(i).Value.TensorNameDesc());
                Inputs[requiredInputs.ElementAt(i).Value.Name] = userInputs[i];
            }
        }
        op.Complete();  
        return true;
    }
    public bool Execute(ITensor[] userInputs, ExecutionProvider provider = ExecutionProvider.CPU)
    {
        if (!ResolveInputs(userInputs))
        {
            return false;
        }

        var op = Begin("Executing graph {n} from {f}", Metadata["Name"], ModelFile);
        foreach (var node in Nodes) 
        {
            var r = node.Execute(this);
            if (r.Status == OpStatus.Failure)
            {
                Error("Execution of node {n} with op {op} failed: {m}.", node.Name, node.Op, r.Message ?? "");
                Error("Stopping graph execution at node {n}.", node.Name);
                return false;
            }
            else
            {
                Debug("Execution of node {n} with op {op} returned {s} with {c} output(s).", node.Name, node.Op, r.Status.ToString(), r.Outputs.Length);
                for (int i = 0; i < node.Outputs.Length; i++)
                {
                    Debug("Assigning graph tensor {o} to node {n} output {c}.", node.Outputs[i], node.Name, i);
                    if (IntermediateOutputs.ContainsKey(node.Outputs[i]))
                    {
                        IntermediateOutputs[node.Outputs[i]] = r.Outputs[i];
                    }
                    else
                    {
                        Outputs[node.Outputs[i]] = r.Outputs[i];
                    }
                }
            }
        }
        op.Complete();
        return true;    
        
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