namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;

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

    public ExecutionOptions Options { get; set; } = ExecutionOptions.Default;

    public Dictionary<string, int> Opset = new Dictionary<string, int>();

    public Dictionary<string, object> Metadata = new Dictionary<string, object>();

    public Dictionary<string, string> MetadataProps = new Dictionary<string, string>();

    public Stack<NodeProfile>? LastProfile { get; private set; }

    public string? LastErrorMessage { get; private set; }

    public string? LastFailedNodeName { get; private set; }

    public OpType? LastFailedNodeOp { get; private set; }

    /// <summary>Last file-order node index consuming each tensor name.</summary>
    public Dictionary<string, int> LastUseIndex { get; private set; } = new Dictionary<string, int>(StringComparer.Ordinal);

    internal TensorBufferPool? ActivePool { get; private set; }

    /// <summary>Pool arrays freshly allocated during the last execution.</summary>
    public int LastPoolAllocatedNew { get; private set; }

    /// <summary>Pool rents served from previously returned arrays during the last execution.</summary>
    public int LastPoolReused { get; private set; }

    /// <summary>Dead dense buffers adopted into the pool during the last execution.</summary>
    public int LastPoolReturned { get; private set; }

    /// <summary>Fresh pool array bytes allocated during the last execution.</summary>
    public long LastPoolAllocatedNewBytes { get; private set; }

    /// <summary>Pooled output bytes served from previously returned arrays during the last execution.</summary>
    public long LastPoolReusedBytes { get; private set; }
    #endregion

    #region Methods
    public int OpsetVersion(string domain = "") => this.Opset.ContainsKey(domain) ? this.Opset[domain] : throw new InvalidOperationException($"The domain {domain} does not ezist in the imported opsets.");

    public ITensor GetInputTensor(string name)
    {
        if (Inputs.TryGetValue(name, out var input)) return input;
        if (Initializers.TryGetValue(name, out var init)) return init;
        if (IntermediateOutputs.TryGetValue(name, out var mid) && mid is not null) return mid;
        if (Outputs.TryGetValue(name, out var output)) return output;
        throw new InvalidOperationException($"The intermediate output tensor {name} has not been assigned a value.");
    }
    public ITensor? GetInputTensor(string[] Inputs, int index) =>
       index < Inputs.Length ? GetInputTensor(Inputs[index]) : null;    

    public ITensor[] GetInputTensors(string[] names) =>
        names.Where(n => !string.IsNullOrEmpty(n)).Select(n => GetInputTensor(n)).ToArray();

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
        }
        return requiredInputs;
    }

    public bool ResolveInputs(ITensor[] userInputs, bool useInitializers)
    {
        var op = Begin("Resolving {c} graph inputs for execution", Inputs.Count);
        var requiredInputs = GetRequiredInputs(useInitializers);   
        Info("{uic} user input(s) required for graph execution: {uig}.", requiredInputs.Count, requiredInputs.Select(ui => ui.Value.TensorNameDesc()));
        if (userInputs.Length != requiredInputs.Count)
        {
            Error("{uic} user input(s) required for graph execution:{i} but only {c} specified.", requiredInputs.Count, userInputs.Select(ui => ui.TensorNameDesc()), userInputs.Length);
            op.Abandon();
            return false;
        }
        for(int i = 0; i < requiredInputs.Keys.Count; i++)
        {
            if (!(userInputs[i].Rank == requiredInputs.ElementAt(i).Value.Rank && userInputs[i].ElementType == requiredInputs.ElementAt(i).Value.ElementType))
            {
                Error("Cannot use user input {ui} for required input {ri}. Tensor type or rank does not match.", userInputs[i].TensorNameDesc(), requiredInputs.ElementAt(i).Value.TensorNameDesc());
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

    public bool ResolveInputs(Dictionary<string, ITensor> userInputs, bool useInitializers)
    {
        var op = Begin("Resolving {c} graph inputs for execution", Inputs.Count);
        var requiredInputs = GetRequiredInputs(useInitializers);
        Info("{uic} user input(s) required for graph execution: {uig}.", requiredInputs.Count, requiredInputs.Select(ui => ui.Value.TensorNameDesc()));
        if (userInputs.Count != requiredInputs.Count)
        {
            Error("{uic} user input(s) required for graph execution:{i} but only {c} specified.", requiredInputs.Count, userInputs.Select(ui => ui.Value.TensorNameDesc()), userInputs.Count);
            op.Abandon();
            return false;
        }
        foreach (var kv in requiredInputs)
        {
            if (!(userInputs[kv.Key].Rank == kv.Value.Rank && userInputs[kv.Key].ElementType == kv.Value.ElementType))
            {
                Error("Cannot use user input {ui} for required input {ri}. Tensor type or rank does not match.", userInputs[kv.Key].TensorNameDesc(), kv.Value.TensorNameDesc());
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

    public bool ResolveNodeExecuteInputs(Node node, ITensor[] userInputs, bool useInitializers)
    {
        var op = Begin("Resolving {c} node inputs for execution", node.Inputs.Length);
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
            Error("{uic} user input(s) required for node execution:{i} but {c} specified.", requiredInputs.Count, userInputs.Select(ui => ui.TensorNameDesc()), userInputs.Length);
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

    public bool ResolveNodeExecuteInputs(Node node, Dictionary<string, ITensor> userInputs, bool useInitializers)
    {
        var op = Begin("Resolving {c} node inputs for execution", node.Inputs.Length);
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
            Error("{uic} user input(s) required for node execution:{i} but {c} specified.", requiredInputs.Count, userInputs.Select(ui => ui.Value.TensorNameDesc()), userInputs.Count);
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

    public bool Execute(object userInputs, bool useInitializers, ExecutionProvider provider = ExecutionProvider.CPU, ExecutionOptions? options = null)
    {
        LastErrorMessage = null;
        LastFailedNodeName = null;
        LastFailedNodeOp = null;
        LastProfile = null;
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
            Error("Unsupported user inputs type: {t}.", userInputs.GetType().Name);
            return false;
        }

        int count = 0;
        var op = Begin("Executing graph {n} from {f}", Metadata["Name"], ModelFile);

        using var profilerScope = Profiler.BeginExecution();
        using var poolScope = new ExecutionPoolScope(this);
        foreach (var node in Nodes)
        {
            count++;
            Debug("Executing node {c} {node} with op: {op}, inputs: {inputs}, outputs: {outputs} and "
                + ((node.Attributes is not null && node.Attributes.Count > 0) ? "the following attributes:" : "no attributes."),
                count, node.Name, node.Op.ToString(),
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
           
            Profiler.StartNodeProfile(node.ID, node.Op);
            var r = node.Execute(this, provider, options ?? Options);
            Profiler.StopNodeProfile();
            
            if (r.Status == OpStatus.Failure)
            {
                Error("Execution of node {c} {n} with op {op} failed: {m}", count, node.Name, node.Op, r.Message ?? "");
                Error("Stopping graph execution at node {c} {n}.", count, node.Name);
                LastErrorMessage = r.Message;
                LastFailedNodeName = node.Name;
                LastFailedNodeOp = node.Op;
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
                ReleaseDeadTensors(node, count - 1);
            }
        }
        LastProfile = profilerScope.Profile;
        op.Complete();
        return true;
    }

    public bool ExecuteNode(object userInputs, string nodeLabel, bool useInitializers, ExecutionProvider provider = ExecutionProvider.CPU, ExecutionOptions? options = null)
    {
        var node = Nodes.FirstOrDefault(n => n.Name == nodeLabel);
        if (node.Name == "")
        {
            Error("Could not find node {n} in graph.", nodeLabel); 
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
        var r = node.Execute(this, provider, options ?? Options);
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
        Info("Reset graph state.");
    }

    /// <summary>Recomputes <see cref="LastUseIndex"/> from the current <see cref="Nodes"/> order.</summary>
    /// <remarks>Graph outputs map to <see cref="Nodes"/>.Count (live to the end); graph inputs and
    /// initializers map to <see cref="int.MaxValue"/> (live forever); produced-but-unconsumed
    /// intermediates map to their producer index. Inert: no execution state changes.</remarks>
    public void RefreshLifetimeAnalysis()
    {
        var lastUse = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < Nodes.Count; i++)
        {
            var inputs = Nodes[i].Inputs;
            if (inputs is null) continue;
            foreach (var input in inputs)
            {
                if (string.IsNullOrEmpty(input)) continue;
                lastUse[input] = i;
            }
        }
        foreach (var name in Outputs.Keys)
        {
            if (!string.IsNullOrEmpty(name)) lastUse[name] = Nodes.Count;
        }
        foreach (var name in Inputs.Keys)
        {
            if (!string.IsNullOrEmpty(name)) lastUse[name] = int.MaxValue;
        }
        foreach (var name in Initializers.Keys)
        {
            if (!string.IsNullOrEmpty(name)) lastUse[name] = int.MaxValue;
        }
        for (int i = 0; i < Nodes.Count; i++)
        {
            var outputs = Nodes[i].Outputs;
            if (outputs is null) continue;
            foreach (var output in outputs)
            {
                if (string.IsNullOrEmpty(output)) continue;
                if (!lastUse.ContainsKey(output)) lastUse[output] = i;
            }
        }
        LastUseIndex = lastUse;
    }

    readonly struct ExecutionPoolScope : IDisposable
    {
        readonly ComputationalGraph graph;
        public ExecutionPoolScope(ComputationalGraph graph)
        {
            this.graph = graph;
            graph.ActivePool = new TensorBufferPool();
        }
        public void Dispose()
        {
            if (graph.ActivePool is not null)
            {
                graph.LastPoolAllocatedNew = graph.ActivePool.AllocatedNew;
                graph.LastPoolReused = graph.ActivePool.Reused;
                graph.LastPoolReturned = graph.ActivePool.Returned;
                graph.LastPoolAllocatedNewBytes = graph.ActivePool.AllocatedNewBytes;
                graph.LastPoolReusedBytes = graph.ActivePool.ReusedBytes;
            }
            graph.ActivePool = null;
        }
    }

    void ReleaseDeadTensors(Node node, int index)
    {
        var pool = ActivePool;
        if (pool is null || node.Inputs is null) return;
        foreach (var name in node.Inputs)
        {
            if (string.IsNullOrEmpty(name)) continue;
            if (!LastUseIndex.TryGetValue(name, out var last) || last != index) continue;
            if (node.Outputs is not null && node.Outputs.Contains(name)) continue;
            if (Outputs.ContainsKey(name)) continue;
            if (!IntermediateOutputs.TryGetValue(name, out var tensor) || tensor is not DenseTensor<float> dense) continue;
            if (!MemoryMarshal.TryGetArray<float>(dense.Buffer, out var segment) || segment.Array is null) continue;
            if (HasLiveAlias(segment.Array, tensor)) continue;
            pool.Return(segment.Array);
            IntermediateOutputs[name] = null;
        }
    }

    bool HasLiveAlias(Array candidate, ITensor self)
    {
        foreach (var tensor in Inputs.Values) if (!ReferenceEquals(tensor, self) && SharesPooledStorage(candidate, tensor)) return true;
        foreach (var tensor in Initializers.Values) if (!ReferenceEquals(tensor, self) && SharesPooledStorage(candidate, tensor)) return true;
        foreach (var tensor in IntermediateOutputs.Values)
        {
            if (tensor is null || ReferenceEquals(tensor, self)) continue;
            if (SharesPooledStorage(candidate, tensor)) return true;
        }
        foreach (var tensor in Outputs.Values) if (!ReferenceEquals(tensor, self) && SharesPooledStorage(candidate, tensor)) return true;
        return false;
    }

    static bool SharesPooledStorage(Array candidate, ITensor tensor)
    {
        if (tensor is Tensor<float> typed)
        {
            try
            {
                return MemoryMarshal.TryGetArray<float>(typed.Storage, out var segment) && ReferenceEquals(segment.Array, candidate);
            }
            catch (NotImplementedException)
            {
                return true;
            }
        }
        return false;
    }
    #endregion
}
