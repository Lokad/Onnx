namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

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

    public Dictionary<string, int> Opset = new Dictionary<string, int>();

    public Dictionary<string, object> Metadata = new Dictionary<string, object>();

    public Dictionary<string, string> MetadataProps = new Dictionary<string, string>();
    #endregion

    #region Methods
    public int OpsetVersion(string domain = "") => this.Opset.ContainsKey(domain) ? this.Opset[domain] : throw new InvalidOperationException($"The domain {domain} does not ezist in the imported opsets.");

    public ITensor GetInputTensor(string name) =>
        Inputs.ContainsKey(name) ? Inputs[name] : Initializers.ContainsKey(name) ? Initializers[name] : 
            IntermediateOutputs[name] ?? throw new InvalidOperationException($"The intermediate output tensor {name} has not been assigned a value.");

    public ITensor? GetInputTensor(string[] Inputs, int index) =>
       index < Inputs.Length ? GetInputTensor(Inputs[index]) : null;    

    public ITensor[] GetInputTensors(string[] names) => names.Select(n => GetInputTensor(n)).ToArray();

    public Dictionary<string, ITensor> GetRequiredInputs(bool useInitializers)
    {
        var requiredInputs = new Dictionary<string, ITensor>(Inputs);
        if (useInitializers)
        {
            foreach (var i in Inputs.Keys)
            {
                if (Initializers.ContainsKey(i))
                {
                    var ii = Initializers[i];
                    var iv = Inputs[i];
                    if (ii.Dims.SequenceEqual(iv.Dims) && ii.ElementType == iv.ElementType)
                    {
                        Inputs[i] = Initializers[i];
                        requiredInputs.Remove(i);
                    }
                }
            }
        }
        return requiredInputs;
    }

    public bool ResolveInputs(ITensor[] userInputs, bool useInitializers)
    {
        var requiredInputs = GetRequiredInputs(useInitializers);   
        if (userInputs.Length != requiredInputs.Count)
        {
            return false;
        }
        for(int i = 0; i < requiredInputs.Keys.Count; i++)
        {
            if (!(userInputs[i].Rank == requiredInputs.ElementAt(i).Value.Rank && userInputs[i].ElementType == requiredInputs.ElementAt(i).Value.ElementType))
            {
                return false;
            }
            else
            {
                Inputs[requiredInputs.Keys.ElementAt(i)] = userInputs[i];
            }
        }
        return true;
    }

    public bool ResolveInputs(Dictionary<string, ITensor> userInputs, bool useInitializers)
    {
        var requiredInputs = GetRequiredInputs(useInitializers);
        if (userInputs.Count != requiredInputs.Count)
        {
            return false;
        }
        foreach (var kv in requiredInputs)
        {
            if (!(userInputs[kv.Key].Rank == kv.Value.Rank && userInputs[kv.Key].ElementType == kv.Value.ElementType))
            {
                return false;
            }
            else
            {
                Inputs[kv.Key] = userInputs[kv.Key];
            }
        }
        return true;
    }

    public bool ResolveNodeExecuteInputs(Node node, ITensor[] userInputs, bool useInitializers)
    {
        var requiredInputs = new List<string>();
        foreach(var i in node.Inputs)
        {
            if (useInitializers && Initializers.ContainsKey(i))
            {
                Inputs[i] = Initializers[i];  
            }
            else
            {
                requiredInputs.Add(i);
            }
        }
        if (userInputs.Length != requiredInputs.Count)
        {
            return false;
        }
        for (int i = 0; i < requiredInputs.Count; i++)
        { 
            Inputs[requiredInputs[i]] = userInputs[i];
            
        }
        return true;
    }

    public bool ResolveNodeExecuteInputs(Node node, Dictionary<string, ITensor> userInputs, bool useInitializers)
    {
        var requiredInputs = new List<string>();
        foreach (var i in node.Inputs)
        {
            if (useInitializers && Initializers.ContainsKey(i))
            {
                Inputs[i] = Initializers[i];
            }
            else
            {
                requiredInputs.Add(i);
            }
        }
        if (userInputs.Count != requiredInputs.Count)
        {
            return false;
        }
        for (int i = 0; i < requiredInputs.Count; i++)
        {
            if (!userInputs.ContainsKey(requiredInputs[i]))
            {
                return false;
            }
            else
            {
                Inputs[requiredInputs[i]] = userInputs[requiredInputs[i]];
            }
        }
        return true;
    }

    public bool Execute(object userInputs, bool useInitializers, ExecutionProvider provider = ExecutionProvider.CPU, int? optimes = -1, bool nodetimes=false)
    {
        if (userInputs is ITensor[] uia)
        {
            if (!ResolveInputs(uia, useInitializers))
            {
                return false;
            }
        }
        else if (userInputs is Dictionary<string, ITensor> uid)
        {
            if (!ResolveInputs(uid, useInitializers))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        Dictionary<OpType, double> opTimes = new Dictionary<OpType, double>();
        Dictionary<OpType, long> opCounts = new Dictionary<OpType, long>();
        int count = 0;
        Stopwatch timer = new Stopwatch(); 
        
        foreach (var node in Nodes) 
        {
            count++;
            if (!opCounts.ContainsKey(node.Op))
            {
                opCounts[node.Op] = 0;
                opTimes[node.Op] = 0L;
            }

            timer.Start();
            var r = node.Execute(this);
            timer.Stop();

            opCounts[node.Op]++;
            var elapsed = timer.Elapsed.TotalMilliseconds;
            opTimes[node.Op] += elapsed;
            timer.Reset();
            if (r.Status == OpStatus.Failure)
            {
                return false;
            }
            else
            {
                for (int i = 0; i < node.Outputs.Length; i++)
                {
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
        return true;    
    }

    public bool ExecuteNode(object userInputs, string nodeLabel, bool useInitializers, ExecutionProvider provider = ExecutionProvider.CPU)
    {
        var node = Nodes.FirstOrDefault(n => n.Name == nodeLabel);
        if (node.Name == "")
        {
            return false;    
        }
        if (userInputs is ITensor[] uia)
        {
            if (!ResolveNodeExecuteInputs(node, uia, useInitializers))
            {
                return false;
            }
        }
        else if (userInputs is Dictionary<string, ITensor> uid)
        {
            if (!ResolveNodeExecuteInputs(node, uid, useInitializers))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        var r = node.Execute(this);
        if (r.Status == OpStatus.Failure)
        {
            return false;
        }
        else
        {
            Outputs.Clear();
            for (int i = 0; i < node.Outputs.Length; i++)
            {
                Outputs[node.Outputs[i]] = r.Outputs[i];
            }
        }
        return true;
    }

    public void Reset(bool gc = false)
    {
        foreach (var o in IntermediateOutputs.Keys)
        {
            IntermediateOutputs[o] = null;
        }
        Outputs = Model.Graph.Output.ToDictionary(vp => vp.Name, vp => vp.ToTensor());
        if (gc)
        {
            GC.Collect(2, GCCollectionMode.Forced, true, true);
            GC.WaitForPendingFinalizers();
        }
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