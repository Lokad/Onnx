namespace Lokad.Onnx;

using Satsuma;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Linq;
using static Lokad.Onnx.Logger;

public class ComputationalGraph : Runtime
{
    #region Fields
    public string ModelFile = "";

    public ModelProto Model = new ModelProto();

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

    public Dictionary<string, ITensor> GetRequiredInputs()
    {
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
                }
            }
        }
        return requiredInputs;
    }

    public bool ResolveInputs(ITensor[] userInputs)
    {
        var op = Begin("Resolving {c} graph inputs for execution", Inputs.Count);
        var requiredInputs = GetRequiredInputs();   
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
                Inputs[requiredInputs.Keys.ElementAt(i)] = userInputs[i];
            }
        }
        op.Complete();  
        return true;
    }

    public bool ResolveInputs(Dictionary<string, ITensor> userInputs)
    {
        var op = Begin("Resolving {c} graph inputs for execution", Inputs.Count);
        var requiredInputs = GetRequiredInputs();
        Info("{uic} user input(s) required for graph execution: {uig}.", requiredInputs.Count, requiredInputs.Select(ui => ui.Value.TensorNameDesc()));
        if (userInputs.Count != requiredInputs.Count)
        {
            Error("{uic} user input(s) required for graph execution:{i} but only {c} specified.", requiredInputs.Count, userInputs.Select(ui => ui.Value.TensorNameDesc()), userInputs.Count);
            op.Abandon();
            return false;
        }
        foreach (var kv in requiredInputs)
        {
            if (!userInputs[kv.Key].Dims.SequenceEqual(kv.Value.Dims) || !(userInputs[kv.Key].ElementType == kv.Value.ElementType))
            {
                Error("Cannot use user input {ui} for required input {ri}. Tensor shape or type does not match.", userInputs[kv.Key].TensorNameDesc(), kv.Value.TensorNameDesc());
                op.Abandon();
                return false;
            }
            else
            {
                Info("Using user input {n} for graph input {i}.", userInputs[kv.Key].TensorNameDesc(), kv.Value.TensorNameDesc());
                Inputs[kv.Key] = userInputs[kv.Key];
            }
        }
        return true;
    }

    public bool ResolveNodeExecuteInputs(Node node, ITensor[] userInputs)
    {
        var op = Begin("Resolving {c} node inputs for execution", node.Inputs.Length);
        var requiredInputs = new List<string>();
        foreach(var i in node.Inputs)
        {
            if (!Initializers.ContainsKey(i))
            {
                requiredInputs.Add(i);  
            }
            else
            {
                Inputs[i] = Initializers[i];
            }
        }
        if (userInputs.Length != requiredInputs.Count)
        {
            Error("{uic} user input(s) required for node execution:{i} but only {c} specified.", requiredInputs.Count, userInputs.Select(ui => ui.TensorNameDesc()), userInputs.Length);
            op.Abandon();
            return false;
        }
        for (int i = 0; i < requiredInputs.Count; i++)
        { 
            Inputs[requiredInputs[i]] = userInputs[i];
            
        }
        op.Complete();
        return true;
    }

    public bool ResolveNodeExecuteInputs(Node node, Dictionary<string, ITensor> userInputs)
    {
        var op = Begin("Resolving {c} node inputs for execution", node.Inputs.Length);
        var requiredInputs = new List<string>();
        foreach (var i in node.Inputs)
        {
            if (!Initializers.ContainsKey(i))
            {
                requiredInputs.Add(i);
            }
            else
            {
                Inputs[i] = Initializers[i];
            }
        }
        if (userInputs.Count != requiredInputs.Count)
        {
            Error("{uic} user input(s) required for node execution:{i} but only {c} specified.", requiredInputs.Count, userInputs.Select(ui => ui.Value.TensorNameDesc()), userInputs.Count);
            op.Abandon();
            return false;
        }
        for (int i = 0; i < requiredInputs.Count; i++)
        {
            if (!userInputs.ContainsKey(requiredInputs[i]))
            {
                Error("User inputs do not contain required input {i}.", requiredInputs[i]);
                return false;
            }
            else
            {
                Inputs[requiredInputs[i]] = userInputs[requiredInputs[i]];
            }
        }
        op.Complete();
        return true;
    }

    public bool Execute(object userInputs, ExecutionProvider provider = ExecutionProvider.CPU)
    {
        if (userInputs is ITensor[] uia)
        {
            if (!ResolveInputs(uia))
            {
                return false;
            }
        }
        else if (userInputs is Dictionary<string, ITensor> uid)
        {
            if (!ResolveInputs(uid))
            {
                return false;
            }
        }
        else
        {
            Error("Unsupported user inputs type: {t}.", userInputs.GetType().Name);
            return false;
        }

        var op = Begin("Executing graph {n} from {f}", Metadata["Name"], ModelFile);
        foreach (var node in Nodes) 
        {
            Debug("Executing node {node} with op: {op}, inputs: {inputs}, outputs: {outputs} and "
                + ((node.Attributes is not null && node.Attributes.Count > 0) ? "the following attributes:" : "no attributes."),
                node.Name, node.Op.ToString(),
                GetInputTensors(node.Inputs).Select(t => t.TensorNameDesc()),
                node.Outputs
            );
            if (node.Attributes is not null && node.Attributes.Count > 0)
            {
                foreach (var kv in node.Attributes)
                {
                    Debug("  {n}: {v}", kv.Key, kv.Value);
                }
            }
            var r = node.Execute(this);
            if (r.Status == OpStatus.Failure)
            {
                Error("Execution of node {n} with op {op} failed: {m}.", node.Name, node.Op, r.Message ?? "");
                Error("Stopping graph execution at node {n}.", node.Name);
                return false;
            }
            else
            {
                Debug("Execution of node {n} with op {op} returned {s} with {c} output(s).", node.Name, node.Op.ToString(), r.Status.ToString(), r.Outputs.Length);
                for (int i = 0; i < node.Outputs.Length; i++)
                {
                    Debug("Assigning node {n} output {c} to graph tensor {o}.", node.Name, i, node.Outputs[i]);
                    if (IntermediateOutputs.ContainsKey(node.Outputs[i]))
                    {
                        IntermediateOutputs[node.Outputs[i]] = r.Outputs[i];
                        r.Outputs[i].Name = node.Outputs[i];
                    }
                    else
                    {
                        Outputs[node.Outputs[i]] = r.Outputs[i];
                        r.Outputs[i].Name = node.Outputs[i];
                    }
                }
            }
        }
        op.Complete();
        return true;    
    }

    public bool ExecuteNode(object userInputs, string nodeLabel, ExecutionProvider provider = ExecutionProvider.CPU)
    {
        var node = Nodes.FirstOrDefault(n => n.Name == nodeLabel);
        if (node.Name == "")
        {
            Error("Could not find node {n} in graph.", nodeLabel); 
            return false;    
        }
        if (userInputs is ITensor[] uia)
        {
            if (!ResolveNodeExecuteInputs(node, uia))
            {
                return false;
            }
        }
        else if (userInputs is Dictionary<string, ITensor> uid)
        {
            if (!ResolveNodeExecuteInputs(node, uid))
            {
                return false;
            }
        }
        else
        {
            Error("Unsupported user inputs type: {t}.", userInputs.GetType().Name);
            return false;
        }

        var op = Begin("Executing node {node} in graph {n} from {f}", nodeLabel, Metadata["Name"], ModelFile);
        Debug("Executing node {node} with op: {op}, inputs: {inputs}, outputs: {outputs} and "
            + ((node.Attributes is not null && node.Attributes.Count > 0) ? "the following attributes:" : "no attributes."),
            node.Name, node.Op.ToString(),
            GetInputTensors(node.Inputs).Select(t => t.TensorNameDesc()),
            node.Outputs
        );
        if (node.Attributes is not null && node.Attributes.Count > 0)
        {
            foreach (var kv in node.Attributes)
            {
                Debug("  {n}: {v}", kv.Key, kv.Value);
            }
        }
        var r = node.Execute(this);
        if (r.Status == OpStatus.Failure)
        {
            Error("Execution of node {n} with op {op} failed: {m}.", node.Name, node.Op, r.Message ?? "");
            return false;
        }
        else
        {
            Debug("Execution of node {n} with op {op} returned {s} with {c} output(s).", node.Name, node.Op.ToString(), r.Status.ToString(), r.Outputs.Length);
            Outputs.Clear();
            for (int i = 0; i < node.Outputs.Length; i++)
            {
                Debug("Assigning node {n} output {c} to graph tensor {o}.", node.Name, i, node.Outputs[i]);
                Outputs[node.Outputs[i]] = r.Outputs[i];
            }
        }
        op.Complete();
        return true;
    }

    public void Reset()
    {
        Inputs = Model.Graph.Input.ToDictionary(vp => vp.Name, vp => vp.ToTensor());
        Outputs = Model.Graph.Output.ToDictionary(vp => vp.Name, vp => vp.ToTensor());
        foreach (var o in IntermediateOutputs.Keys)
        {
            IntermediateOutputs[o] = null;
        }
        Info("Reset graph state.");
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