namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using static Lokad.Onnx.Runtime;

public class ComputationalGraph
{
    #region Fields
    public string ModelFile = "";

    public BindingMap Inputs = new BindingMap();

    public BindingMap Outputs = new BindingMap();

    public List<OnnxValueInfo> OutputDescs = new List<OnnxValueInfo>();

    public Dictionary<string, ITensor> Initializers = new Dictionary<string, ITensor>();

    /// <summary>
    /// A constant-only Transpose folded into a prepared initializer. The
    /// transposed payload is owned immutably by the prepared plan: replacing
    /// the source initializer is detected per run and refolds on demand,
    /// while in-place content mutation requires <see cref="InvalidatePreparation"/>.
    /// </summary>
    internal sealed record FoldedTranspose(string SourceName, ITensor SourceRef, long SourceLength, string PreparedName);

    internal Dictionary<string, FoldedTranspose> FoldedTransposes = new Dictionary<string, FoldedTranspose>(StringComparer.Ordinal);

    internal object FoldLock = new object();

    internal object PrepareLock = new object();

    /// <summary>
    /// Retained input descriptions (with symbolic dimension names) for
    /// descriptor-aware validation. Treated as immutable after load; empty for
    /// hand-built graphs, which validate against placeholder shapes instead.
    /// </summary>
    public List<OnnxValueInfo> InputDescs = new List<OnnxValueInfo>();

    public Dictionary<string, ITensor?> IntermediateOutputs = new Dictionary<string, ITensor?>();

    public List<Node> Nodes { get; set; } = new List<Node>();

    public ExecutionOptions Options { get; set; } = ExecutionOptions.Default;

    public Dictionary<string, int> Opset = new Dictionary<string, int>();

    public Dictionary<string, object> Metadata = new Dictionary<string, object>();

    public Dictionary<string, string> MetadataProps = new Dictionary<string, string>();

    public Stack<NodeProfile>? LastProfile { get; private set; }

    /// <summary>Wall time of the last run body on this instance.</summary>
    /// <remarks>
    /// Covers binding, node execution, and release, but not preparation,
    /// which runs before the body. Partial runs record time-to-exit with a
    /// null profile, matching the existing null-profile-on-early-failure
    /// behavior. Published back to the facade like every run diagnostic.
    /// </remarks>
    public TimeSpan LastRunTime { get; private set; }

    /// <summary>Managed bytes allocated on the calling thread during the last run body.</summary>
    /// <remarks>
    /// Exact for single-threaded runs; a lower bound when worker threads
    /// allocate. Covers binding, node execution, and release like LastRunTime,
    /// but not preparation, which runs before the body.
    /// </remarks>
    public long LastAllocatedBytes { get; private set; }

    /// <summary>Per-generation GC collection deltas observed around the last run body.</summary>
    /// <remarks>
    /// Fresh 3-length array per stamp; index is the generation. Deltas come
    /// from process-wide collector counters, so concurrent suites can advance
    /// them: they bound collections during the run, never attribute them.
    /// </remarks>
    public int[] LastGcCollections { get; private set; } = new int[3];

    public string? LastErrorMessage { get; private set; }

    public string? LastFailedNodeName { get; private set; }

    public OpType? LastFailedNodeOp { get; private set; }

    /// <summary>Original exception behind the last failure, when one was captured; null otherwise.</summary>
    public Exception? LastErrorCause { get; private set; }

    /// <summary>Last file-order node index consuming each tensor name.</summary>
    public Dictionary<string, int> LastUseIndex { get; internal set; } = new Dictionary<string, int>(StringComparer.Ordinal);

    internal TensorBufferPool? ActivePool { get; private set; }

    internal IScratchAccountant? ActiveScratch { get; private set; }

    internal ICopyAccountant? ActiveCopy { get; private set; }

    /// <summary>
    /// Live reference counts per backing array over the variable bindings
    /// (intermediates plus outputs), null until the first release probe of a
    /// run seeds them. Missing counts mean no live user, never the reverse.
    /// </summary>
    Dictionary<Array, int>? liveArrayUsers;

    /// <summary>
    /// Live tensors whose roots escape traversal, held by reference. Any
    /// member forces the conservative answer until it is dropped.
    /// </summary>
    HashSet<ITensor>? liveUnknownTensors;

    bool liveIndexSeeded;

    /// <summary>Reentrancy guard: at most one execution at a time per graph or context.</summary>
    protected int _executing;
    /// <summary>Whether lifetime analysis is current for <see cref="Nodes"/>.</summary>
    protected bool _prepared;
    protected long _preparedFingerprint;
    protected string? _preparationError;

    /// <summary>
    /// Recomputes lifetime analysis for the current <see cref="Nodes"/> order.
    /// Execution entries re-analyze automatically when the structure changes,
    /// so an explicit call only warms the analysis up before the first run.
    /// </summary>
    /// <remarks>
    /// Legal mutation rule: preparation is refused while the same instance is
    /// executing. Direct edits to the public <see cref="Nodes"/> list during a
    /// run remain caller error requiring external synchronization.
    /// </remarks>
    public void Prepare()
    {
        if (System.Threading.Volatile.Read(ref _executing) != 0)
        {
            throw new InvalidOperationException("Graph preparation is not allowed while the graph is executing.");
        }
        lock (PrepareLock)
        {
            if (System.Threading.Volatile.Read(ref _executing) != 0)
            {
                throw new InvalidOperationException("Graph preparation is not allowed while the graph is executing.");
            }
            RefreshLifetimeAnalysis();
        }
    }

    /// <summary>Forgets the analysis so the next execution re-analyzes unconditionally.</summary>
    /// <remarks>Drops folded constant transposes as well, so in-place content
    /// mutation of a folded initializer takes effect after this call.</remarks>
    public void InvalidatePreparation()
    {
        if (System.Threading.Volatile.Read(ref _executing) != 0)
        {
            throw new InvalidOperationException("Graph preparation is not allowed while the graph is executing.");
        }
        lock (PrepareLock)
        {
            if (System.Threading.Volatile.Read(ref _executing) != 0)
            {
                throw new InvalidOperationException("Graph preparation is not allowed while the graph is executing.");
            }
            _prepared = false;
            lock (FoldLock)
            {
                foreach (var fold in FoldedTransposes.Values) Initializers.Remove(fold.PreparedName);
                FoldedTransposes.Clear();
            }
        }
    }

    /// <summary>
    /// Returns the prepared transposition for a constant-only Transpose node
    /// when the source initializer is unchanged since the fold. Any doubt
    /// falls through to normal computation.
    /// </summary>
    internal bool TryGetFoldedTranspose(string nodeName, string[]? nodeInputs, ITensor? current, out ITensor? prepared)
    {
        prepared = null;
        if (string.IsNullOrEmpty(nodeName) || nodeInputs is null || nodeInputs.Length != 1
            || string.IsNullOrEmpty(nodeInputs[0]) || current is null) return false;
        lock (FoldLock)
        {
            if (!FoldedTransposes.TryGetValue(nodeName, out var fold)) return false;
            if (fold.SourceName != nodeInputs[0]) return false;
            if (!Initializers.TryGetValue(fold.SourceName, out var src)
                || !ReferenceEquals(src, fold.SourceRef) || src.Length != fold.SourceLength) return false;
            if (!ReferenceEquals(current, src)) return false;
            if (!Initializers.TryGetValue(fold.PreparedName, out prepared) || prepared is null) return false;
            return true;
        }
    }

    /// <summary>
    /// Folds a freshly computed constant-only transposition into a prepared
    /// initializer owned by the plan. The stored payload is an independent
    /// copy, never pool-rented storage.
    /// </summary>
    internal void FoldTranspose(string nodeName, string[]? nodeInputs, ITensor? input, ITensor result)
    {
        if (string.IsNullOrEmpty(nodeName) || nodeInputs is null || nodeInputs.Length != 1
            || string.IsNullOrEmpty(nodeInputs[0]) || input is null) return;
        string sourceName = nodeInputs[0];
        lock (FoldLock)
        {
            if (FoldedTransposes.TryGetValue(nodeName, out var existing))
            {
                if (Initializers.TryGetValue(existing.SourceName, out var cur)
                    && ReferenceEquals(cur, existing.SourceRef) && cur.Length == existing.SourceLength) return;
                Initializers.Remove(existing.PreparedName);
                FoldedTransposes.Remove(nodeName);
            }
            if (!Initializers.TryGetValue(sourceName, out var src) || !ReferenceEquals(input, src)) return;
            string preparedName = "folded:" + nodeName;
            if (Initializers.ContainsKey(preparedName) || Inputs.ContainsKey(preparedName)) return;
            var stored = result.Clone();
            stored.Name = preparedName;
            Initializers[preparedName] = stored;
            FoldedTransposes[nodeName] = new FoldedTranspose(sourceName, src, src.Length, preparedName);
        }
    }

    /// <summary>Creates an isolated execution context sharing this prepared plan.</summary>
    public GraphExecution CreateExecution(ExecutionOptions? options)
    {
        lock (PrepareLock)
        {
            EnsurePreparedLocked();
            var exec = new GraphExecution(this, options, _prepared, _preparedFingerprint, _preparationError);
            return exec;
        }
    }

    protected void EnsurePrepared()
    {
        lock (PrepareLock)
        {
            EnsurePreparedLocked();
        }
    }

    void EnsurePreparedLocked()
    {
        if (!_prepared || ComputeStructureFingerprint() != _preparedFingerprint)
        {
            RefreshLifetimeAnalysis();
        }
    }

    void CopyFromExecution(GraphExecution exec)
    {
        lock (PrepareLock)
        {
            ReplaceMap(Inputs, exec.Inputs);
            ReplaceMap(Outputs, exec.Outputs);
            IntermediateOutputs.Clear();
            foreach (var kv in exec.IntermediateOutputs) IntermediateOutputs.Add(kv.Key, kv.Value);
            LastErrorMessage = exec.LastErrorMessage;
            LastFailedNodeName = exec.LastFailedNodeName;
            LastFailedNodeOp = exec.LastFailedNodeOp;
            LastErrorCause = exec.LastErrorCause;
            LastProfile = exec.LastProfile;
            LastRunTime = exec.LastRunTime;
            LastAllocatedBytes = exec.LastAllocatedBytes;
            LastGcCollections = (int[])exec.LastGcCollections.Clone();
            LastPoolAllocatedNew = exec.LastPoolAllocatedNew;
            LastPoolReused = exec.LastPoolReused;
            LastPoolReturned = exec.LastPoolReturned;
            LastPoolDropped = exec.LastPoolDropped;
            LastPoolAllocatedNewBytes = exec.LastPoolAllocatedNewBytes;
            LastPoolReusedBytes = exec.LastPoolReusedBytes;
            LastScratchBytes = exec.LastScratchBytes;
            LastCopyBytes = exec.LastCopyBytes;
            LastPeakLiveBytes = exec.LastPeakLiveBytes;
            LastPoolPeakOutstandingBytes = exec.LastPoolPeakOutstandingBytes;
        }
    }

    static void ReplaceMap(Dictionary<string, ITensor?> target, Dictionary<string, ITensor?> source)
    {
        target.Clear();
        foreach (var kv in source) target.Add(kv.Key, kv.Value);
    }

    /// <summary>Pool arrays freshly allocated during the last execution.</summary>
    public int LastPoolAllocatedNew { get; private set; }

    /// <summary>Pool rents served from previously returned arrays during the last execution.</summary>
    public int LastPoolReused { get; private set; }

    /// <summary>Dead dense buffers adopted into the pool during the last execution.</summary>
    /// <remarks>GC bytes are reported by the collector; this counts pool adoptions, not live payload.</remarks>
    public int LastPoolReturned { get; private set; }

    /// <summary>Pool returns discarded by the per-shape cap during the last execution.</summary>
    /// <remarks>Dropped storage was not retained for reuse; it is reclaimed by the collector.</remarks>
    public int LastPoolDropped { get; private set; }

    /// <summary>Fresh pool array bytes allocated during the last execution.</summary>
    /// <remarks>New GC allocation; pool-served bytes are reported separately via LastPoolReusedBytes.</remarks>
    public long LastPoolAllocatedNewBytes { get; private set; }

    /// <summary>Pooled output bytes served from previously returned arrays during the last execution.</summary>
    /// <remarks>Pool-served bytes avoid new GC allocation; they are not live payload or scratch memory.</remarks>
    public long LastPoolReusedBytes { get; private set; }
    /// <summary>Tensor-copy bytes materialized during the last execution.</summary>
    /// <remarks>RequireContiguous view materialization (MatMul operand and destination
    /// preparation). Distinct from kernel arithmetic writes, concat assembly, GC allocation,
    /// live payload, pool-served bytes, and scratch rents. Single-node runs do not count.</remarks>
    public long LastCopyBytes { get; private set; }
    /// <summary>Kernel scratch bytes rented during the last execution.</summary>
    /// <remarks>Transient im2col patches and GEMM panel packing rented from ArrayPool,
    /// counted at rent time on full graph runs (single-node runs do not count).
    /// Distinct from GC allocation, live payload, and pool-served bytes.</remarks>
    public long LastScratchBytes { get; private set; }
    /// <summary>High-water mark of live logical tensor bytes during the last execution.</summary>
    /// <remarks>Sampled over run bindings (inputs, outputs, live intermediates) before the run,
    /// after every node, and at completion; aliased storage counts per binding, matching the
    /// manual census. Single-node runs do not sample.</remarks>
    public long LastPeakLiveBytes { get; private set; }
    /// <summary>High-water mark of pool bytes checked out during the last execution.</summary>
    /// <remarks>Pool outputs and live intermediates raise it; returns lower it. ArrayPool scratch is not counted.</remarks>
    public long LastPoolPeakOutstandingBytes { get; private set; }
    #endregion

    #region Methods
    public int OpsetVersion(string domain) => this.Opset.ContainsKey(domain) ? this.Opset[domain] : throw new InvalidOperationException($"The domain {domain} does not ezist in the imported opsets.");

    public ITensor GetInputTensor(string name)
    {
        if (Inputs.TryGetValue(name, out var input) && input is not null) return input;
        if (Initializers.TryGetValue(name, out var init)) return init;
        if (IntermediateOutputs.TryGetValue(name, out var mid) && mid is not null) return mid;
        if (Outputs.TryGetValue(name, out var output) && output is not null) return output;
        if (Inputs.ContainsKey(name) || Outputs.ContainsKey(name)) throw new InvalidOperationException($"The graph value {name} is declared but has no value in this run.");
        throw new InvalidOperationException($"The intermediate output tensor {name} has not been assigned a value.");
    }
    public ITensor? GetInputTensor(string[] Inputs, int index) =>
       index < Inputs.Length ? GetInputTensor(Inputs[index]) : null;    

    public ITensor[] GetInputTensors(string[]? names)
    {
        ArgumentNullException.ThrowIfNull(names);
        int count = 0;
        for (int i = 0; i < names.Length; i++)
        {
            if (!string.IsNullOrEmpty(names[i])) count++;
        }
        var resolved = new ITensor[count];
        int next = 0;
        for (int i = 0; i < names.Length; i++)
        {
            if (string.IsNullOrEmpty(names[i])) continue;
            resolved[next++] = GetInputTensor(names[i]);
        }
        return resolved;
    }

    /// <summary>
    /// Returns the retained input descriptor for a graph input name, or null
    /// for hand-built graphs, which validate against placeholder shapes instead.
    /// </summary>
    OnnxValueInfo? FindInputDesc(string name)
    {
        for (int i = 0; i < InputDescs.Count; i++)
        {
            if (InputDescs[i].Name == name) return InputDescs[i];
        }
        return null;
    }

    /// <summary>
    /// Validates one user tensor against a retained descriptor: rank and type
    /// must match, fixed dims (no symbolic name) must match exactly including
    /// real zero extents, anonymous unknown dims (negative sentinel) accept any
    /// non-negative extent without coupling, and symbolic dims accept any
    /// non-negative extent while enforcing agreement across inputs sharing a
    /// symbolic name.
    /// </summary>
    static bool CheckDescriptorDims(OnnxValueInfo desc, ITensor actual, Dictionary<string, int> symbolic, out string? message)
    {
        message = null;
        // Sequence-typed descriptors accept sequence values (of any length)
        // and reject tensors; lengths are irrelevant, only sequence-ness matters.
        if (actual is TensorSequence)
        {
            if (desc.ElementType != TensorElementType.Sequence)
            {
                message = $"Tensor {actual.Name} is a sequence but descriptor {desc.Describe()} is not.";
                return false;
            }
            return true;
        }
        if (desc.Dims.Length != actual.Rank || desc.ElementType != actual.ElementType)
        {
            message = $"Tensor type or rank does not match descriptor {desc.Describe()}.";
            return false;
        }
        var ad = actual.Dims;
        for (int i = 0; i < desc.Dims.Length; i++)
        {
            if (ad[i] < 0)
            {
                message = $"Tensor dimension {ad[i]} is negative.";
                return false;
            }
            string? param = (desc.DimParams is not null && i < desc.DimParams.Length) ? desc.DimParams[i] : null;
            if (param is null)
            {
                if (desc.Dims[i] < 0) continue;
                if (desc.Dims[i] != ad[i])
                {
                    message = $"Tensor dimension {ad[i]} does not match declared fixed dimension {desc.Dims[i]} of {desc.Describe()}.";
                    return false;
                }
            }
            else if (symbolic.TryGetValue(param, out var size))
            {
                if (size != ad[i])
                {
                    message = $"Shared symbolic dimension {param} has conflicting extents {size} and {ad[i]}.";
                    return false;
                }
            }
            else
            {
                symbolic[param] = ad[i];
            }
        }
        return true;
    }

    static bool InputDimsCompatible(ITensor declared, ITensor actual)
    {
        if (declared.Rank != actual.Rank || declared.ElementType != actual.ElementType) return false;
        var dd = declared.Dims;
        var ad = actual.Dims;
        for (int i = 0; i < dd.Length; i++)
        {
            if (ad[i] < 0) return false;
            if (dd[i] > 0 && dd[i] != ad[i]) return false;
        }
        return true;
    }

    /// <summary>
    /// Describes a bound input value for logs, falling back to the retained
    /// input declaration and then the bare name when the slot is unbound.
    /// </summary>
    string DescribeBoundInput(string name)
    {
        if (Inputs.TryGetValue(name, out var bound) && bound is not null) return bound.TensorNameDesc();
        return FindInputDesc(name)?.Describe() ?? name;
    }

    public Dictionary<string, ITensor?> GetRequiredInputs(bool useInitializers)
    {
        var requiredInputs = new Dictionary<string, ITensor?>(Inputs);
        if (useInitializers)
        {
            foreach (var i in Inputs.Keys)
            {
                if (InitializerBacksInput(i)) requiredInputs.Remove(i);
            }
        }
        return requiredInputs;
    }

    /// <summary>
    /// Pure check whether the stored initializer can back input <paramref name="i"/>.
    /// Never mutates bindings; <see cref="ApplyInitializerBindings"/> publishes the
    /// substitution after successful validation.
    /// </summary>
    bool InitializerBacksInput(string i)
    {
        if (!Initializers.TryGetValue(i, out var ii)) return false;
        var declared = FindInputDesc(i);
        Inputs.TryGetValue(i, out var bound);
        var dims = bound?.Dims ?? declared?.Dims;
        var elementType = bound?.ElementType ?? declared?.ElementType;
        if (dims is not null && elementType is not null && ii.Dims.SequenceEqual(dims) && ii.ElementType == elementType) return true;
        if (Log.IsEnabled(LogLevel.Error))
            Error("Cannot use initializer value {n} for graph input {i}. Tensor shape or type does not match.", ii.TensorNameDesc(), DescribeBoundInput(i));
        return false;
    }

    /// <summary>
    /// Publishes every initializer-backed substitution. Call only after input
    /// validation succeeds so a rejected run leaves bindings untouched.
    /// </summary>
    void ApplyInitializerBindings(Dictionary<string, ITensor?> requiredInputs)
    {
        foreach (var i in Inputs.Keys.ToArray())
        {
            if (!requiredInputs.ContainsKey(i) && Initializers.ContainsKey(i))
            {
                Inputs[i] = Initializers[i];
                if (Log.IsEnabled(LogLevel.Info)) Info("Using initializer value {n} for graph input {i}.", Initializers[i].TensorNameDesc(), DescribeBoundInput(i));
            }
        }
    }

    /// <summary>
    /// Scans one node inputs for the names the caller must supply: absent
    /// optional slots bind nothing and repeated inputs resolve once, while
    /// initializer-backed names pre-bind from stored initializers.
    /// </summary>
    List<string> NodeRequiredInputs(Node node, bool useInitializers)
    {
        var requiredInputs = new List<string>();
        foreach (var i in node.Inputs)
        {
            // Absent optional slots bind nothing; repeated inputs resolve once.
            if (string.IsNullOrEmpty(i)) continue;
            if (useInitializers && Initializers.ContainsKey(i))
            {
                Inputs[i] = Initializers[i];
            }
            else if (!requiredInputs.Contains(i))
            {
                requiredInputs.Add(i);
            }
        }
        return requiredInputs;
    }

    /// <summary>
    /// Records an input-count rejection with the same wording for graph and
    /// node binding, describing the supplied values exactly as the callers did.
    /// </summary>
    bool FailInputCountMismatch(string kind, int requiredCount, IEnumerable<string> providedDescs, int providedCount) =>
        Fail("{uic} user input(s) required for " + kind + " execution:{i} but only {c} specified.", requiredCount, providedDescs, providedCount);

    /// <summary>
    /// Checks one supplied tensor against its retained descriptor when one
    /// exists, else against the declared placeholder shape and type.
    /// </summary>
    bool CheckOneBoundInput(string key, ITensor? declared, ITensor provided, Dictionary<string, int> symbolic, out string? detail)
    {
        var desc = FindInputDesc(key);
        detail = null;
        return desc is null
            ? declared is not null && InputDimsCompatible(declared, provided)
            : CheckDescriptorDims(desc, provided, symbolic, out detail);
    }

    public bool ResolveInputs(ITensor[] userInputs, bool useInitializers)
    {
        using var op = Begin("Resolving {c} graph inputs for execution", Inputs.Count);
        var requiredInputs = GetRequiredInputs(useInitializers);   
        Info("{uic} user input(s) required for graph execution: {uig}.", requiredInputs.Count, requiredInputs.Select(ui => DescribeBoundInput(ui.Key)));
        if (userInputs.Length != requiredInputs.Count)
        {
            op.Abandon();
            return FailInputCountMismatch("graph", requiredInputs.Count, userInputs.Select(ui => ui.TensorNameDesc()), userInputs.Length);
        }
        var symbolic = new Dictionary<string, int>(StringComparer.Ordinal);
        var validated = new List<(string Key, ITensor Value)>();
        for(int i = 0; i < requiredInputs.Keys.Count; i++)
        {
            var key = requiredInputs.Keys.ElementAt(i);
            var declared = requiredInputs[key];
            bool ok = CheckOneBoundInput(key, declared, userInputs[i], symbolic, out string? detail);
            if (!ok)
            {
                op.Abandon();
                return Fail("Cannot use user input {ui} for required input {ri}. Tensor type, rank or dimensions do not match. {d}", userInputs[i].TensorNameDesc(), DescribeBoundInput(key), detail ?? "");
            }
            else
            {
                validated.Add((key, userInputs[i]));
            }
        }
        foreach (var (key, value) in validated)
        {
            Info("Using user input {n} for graph input {i}.", value.TensorNameDesc(), DescribeBoundInput(key));
            Inputs[key] = value;
        }
        ApplyInitializerBindings(requiredInputs);
        op.Complete();  
        return true;
    }

    public bool ResolveInputs(Dictionary<string, ITensor> userInputs, bool useInitializers)
    {
        using var op = Begin("Resolving {c} graph inputs for execution", Inputs.Count);
        var requiredInputs = GetRequiredInputs(useInitializers);
        Info("{uic} user input(s) required for graph execution: {uig}.", requiredInputs.Count, requiredInputs.Select(ui => DescribeBoundInput(ui.Key)));
        // Validate names before indexing the user dictionary so unknown or
        // missing names fail cleanly instead of throwing KeyNotFoundException,
        // and validate everything before mutating run state.
        foreach (var name in userInputs.Keys)
        {
            if (!requiredInputs.ContainsKey(name))
            {
                op.Abandon();
                return Fail("User input {n} is not a required graph input.", name);
            }
        }
        var symbolic = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var kv in requiredInputs)
        {
            if (!userInputs.ContainsKey(kv.Key))
            {
                op.Abandon();
                return Fail("User inputs do not contain required input {i}.", kv.Key);
            }
            bool ok = CheckOneBoundInput(kv.Key, kv.Value, userInputs[kv.Key], symbolic, out string? detail);
            if (!ok)
            {
                op.Abandon();
                return Fail("Cannot use user input {ui} for required input {ri}. Tensor type, rank or dimensions do not match. {d}", userInputs[kv.Key].TensorNameDesc(), DescribeBoundInput(kv.Key), detail ?? "");
            }
        }
        foreach (var kv in requiredInputs)
        {
            Info("Using user input {n} for graph input {i}.", userInputs[kv.Key].TensorNameDesc(), DescribeBoundInput(kv.Key));
            Inputs[kv.Key] = userInputs[kv.Key];
        }
        ApplyInitializerBindings(requiredInputs);
        op.Complete();
        return true;
    }

    public bool ResolveNodeExecuteInputs(Node node, ITensor[] userInputs, bool useInitializers)
    {
        using var op = Begin("Resolving {c} node inputs for execution", node.Inputs.Length);
        var requiredInputs = NodeRequiredInputs(node, useInitializers);
        if (userInputs.Length != requiredInputs.Count)
        {
            op.Abandon();
            return FailInputCountMismatch("node", requiredInputs.Count, userInputs.Select(ui => ui.TensorNameDesc()), userInputs.Length);
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
        using var op = Begin("Resolving {c} node inputs for execution", node.Inputs.Length);
        var requiredInputs = NodeRequiredInputs(node, useInitializers);
        if (userInputs.Count != requiredInputs.Count)
        {
            op.Abandon();
            return FailInputCountMismatch("node", requiredInputs.Count, userInputs.Select(ui => ui.Value.TensorNameDesc()), userInputs.Count);
        }
        for (int i = 0; i < requiredInputs.Count; i++)
        {
            if (!userInputs.ContainsKey(requiredInputs[i]))
            {
                op.Abandon();
                return Fail("User inputs do not contain required input {i}.", requiredInputs[i]);
            }
            else
            {
                Inputs[requiredInputs[i]] = userInputs[requiredInputs[i]];
            }
        }
        op.Complete();
        return true;
    }

    public bool Execute(object userInputs, bool useInitializers) => Execute(userInputs, useInitializers, ExecutionProvider.CPU, null);

    /// <summary>
    /// Prepares the graph if needed, binds <paramref name="userInputs"/> (a positional
    /// tensor array or a name-to-tensor map), runs every node with the given provider, and
    /// publishes results on Outputs and IntermediateOutputs. Only one execution may run at
    /// a time; a concurrent call fails. Null <paramref name="options"/> reuses the graph
    /// prepared options.
    /// </summary>
    /// <param name="userInputs">Caller-owned input tensors; the graph never takes ownership of their storage.</param>
    /// <param name="useInitializers">Bind stored initializers for graph inputs left unspecified.</param>
    /// <param name="provider">Execution provider carrying out the operators.</param>
    /// <param name="options">Execution options, or null for the graph prepared options.</param>
    /// <returns>True on success; otherwise false with details on LastErrorMessage, LastFailedNodeName, LastFailedNodeOp and LastErrorCause.</returns>
    public virtual bool Execute(object userInputs, bool useInitializers, ExecutionProvider provider, ExecutionOptions? options)
    {
        if (Interlocked.CompareExchange(ref _executing, 1, 0) != 0)
        {
            return false;
        }
        try
        {
            GraphExecution exec;
            lock (PrepareLock)
            {
                EnsurePreparedLocked();
                exec = new GraphExecution(this, options, _prepared, _preparedFingerprint, _preparationError);
            }
            bool ok = exec.RunCore(userInputs, useInitializers, provider);
            CopyFromExecution(exec);
            return ok;
        }
        finally { Volatile.Write(ref _executing, 0); }
    }

    protected bool RunCore(object userInputs, bool useInitializers, ExecutionProvider provider)
    {
        ResetRunDiagnostics();
        long start = Stopwatch.GetTimestamp();
        long allocStart = GC.GetAllocatedBytesForCurrentThread();
        int gc0 = GC.CollectionCount(0);
        int gc1 = GC.CollectionCount(1);
        int gc2 = GC.CollectionCount(2);
        try
        {
            return RunCoreInner(userInputs, useInitializers, provider);
        }
        finally
        {
            LastRunTime = Stopwatch.GetElapsedTime(start);
            LastAllocatedBytes = GC.GetAllocatedBytesForCurrentThread() - allocStart;
            LastGcCollections = new int[] { GC.CollectionCount(0) - gc0, GC.CollectionCount(1) - gc1, GC.CollectionCount(2) - gc2 };
        }
    }

    bool RunCoreInner(object userInputs, bool useInitializers, ExecutionProvider provider)
    {
        if (!ValidateRunOptions()) return false;
        if (_preparationError is not null) return Fail(_preparationError);
        SeedDeclaredOutputs();
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
            return Fail("Unsupported user inputs type: {t}.", userInputs.GetType().Name);
        }

        int count = 0;
        var boundThisRun = new HashSet<string>(StringComparer.Ordinal);
        List<string>? pendingRelease = null;
        using var op = Begin("Executing graph {n} from {f}", Metadata["Name"], ModelFile);

        using var profilerScope = Profiler.BeginExecution();
        using var poolScope = new ExecutionPoolScope(this);
        var nodeOptions = ActiveScratch is null ? Options : Options with { Tensor = Options.Tensor with { ScratchReporter = ActiveScratch, CopyReporter = ActiveCopy } };
        NoteLivePeak();
        foreach (var node in Nodes)
        {
            count++;
            if (Log.IsEnabled(LogLevel.Debug)) Debug("Executing node {c} {node} with op: {op}, inputs: {inputs}, outputs: {outputs} and "
                + ((node.Attributes is not null && node.Attributes.Count > 0) ? "the following attributes:" : "no attributes."),
                count, node.Name, node.Op.ToString(),
                GetInputTensors(node.Inputs).Select(t => t.TensorNameDesc()),
                node.Outputs
            );
            if (Log.IsEnabled(LogLevel.Debug) && node.Attributes is not null && node.Attributes.Count > 0)
            {
                foreach (var kv in node.Attributes)
                {
                    Debug("  {n}: {v}", kv.Key, kv.Value);
                }
            }
           
            OpResult r;
            try
            {
                Profiler.StartNodeProfile(node.ID, node.Op, () => string.Join(",", GetInputTensors(node.Inputs).Select(t => t.TensorNameDesc())));
                try
                {
                    r = node.Execute(this, provider, nodeOptions);
                }
                finally
                {
                    Profiler.StopNodeProfile();
                }
            }
            catch (Exception ex)
            {
                FailNode(node, ex, "Execution of node {c} {n} with op {op} threw {t}: {m}", count, node.Name, node.Op, ex.GetType().Name, ex.Message);
                throw;
            }

            if (r.Status == OpStatus.Failure)
            {
                FailNode(node, r.Cause, "Execution of node {c} {n} with op {op} failed: {m}", count, node.Name, node.Op, r.Message ?? "");
                Error("Stopping graph execution at node {c} {n}.", count, node.Name);
                return false;
            }
            else
            {
                if (Log.IsEnabled(LogLevel.Debug)) Debug("Execution of node {n} with op {op} returned {s} with {c} output(s).", node.Name, node.Op.ToString(), r.Status.ToString(), r.Outputs.Length);
                for (int i = 0; i < node.Outputs.Length; i++)
                {
                    // Optional outputs use empty names: the value is produced
                    // (positions still validate) but binds to no tensor name.
                    if (string.IsNullOrEmpty(node.Outputs[i])) continue;
                    if (Log.IsEnabled(LogLevel.Debug)) Debug("Assigning node {n} output {c} to graph tensor {o}.", node.Name, i, node.Outputs[i]);
                    if (IntermediateOutputs.ContainsKey(node.Outputs[i]))
                    {
                        TrackBind(IntermediateOutputs, node.Outputs[i], r.Outputs[i]);
                        r.Outputs[i].Name = node.Outputs[i];
                    }
                    else
                    {
                        TrackBind(Outputs, node.Outputs[i], r.Outputs[i]);
                        r.Outputs[i].Name = node.Outputs[i];
                        boundThisRun.Add(node.Outputs[i]);
                    }
                }
                ReleaseDeadTensors(node, count - 1, ref pendingRelease);
                NoteLivePeak();
            }
        }
        // Resolve graph outputs independently of producers: outputs routed
        // to IntermediateOutputs surface here, and outputs aliasing inputs or
        // initializers with no producing node bind those tensors (cloned) rather
        // than leaking zero placeholders. Names bound by nodes win as-is.
        foreach (var name in Outputs.Keys.ToArray())
        {
            if (string.IsNullOrEmpty(name) || boundThisRun.Contains(name)) continue;
            if (IntermediateOutputs.TryGetValue(name, out var mid) && mid is not null)
            {
                TrackBind(Outputs, name, mid);
            }
            else if (Inputs.TryGetValue(name, out var inp) && inp is not null)
            {
                TrackBind(Outputs, name, inp.Clone());
            }
            else if (Initializers.TryGetValue(name, out var init))
            {
                TrackBind(Outputs, name, init.Clone());
            }
        }
        var outputSymbolic = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var desc in OutputDescs)
        {
            if (string.IsNullOrEmpty(desc.Name)) continue;
            if (!Outputs.TryGetValue(desc.Name, out var bound) || bound is null)
            {
                return Fail("Graph output {n} was not resolved by this run.", desc.Name);
            }
            if (!CheckDescriptorDims(desc, bound, outputSymbolic, out var outputDetail))
            {
                return Fail("Graph output {n} does not match its descriptor. {d}", desc.Name, outputDetail ?? "");
            }
        }
        NoteLivePeak();
        LastProfile = profilerScope.Profile;
        op.Complete();
        return true;
    }

    /// <summary>
    /// Re-seeds the output bindings from the immutable output declarations so
    /// every run resolves names even after a failure cleared the map. Declared
    /// keys get fresh null markers; anything else is left untouched.
    /// </summary>
    void SeedDeclaredOutputs()
    {
        foreach (var vp in OutputDescs)
        {
            if (string.IsNullOrEmpty(vp.Name)) continue;
            Outputs.MarkUnresolved(vp.Name);
        }
    }

    /// <summary>
    /// Drops run outputs so a failed execution never leaves previous-run values
    /// looking current. Inputs (descriptors and bindings) are kept so the caller
    /// can correct the inputs and retry.
    /// </summary>
    /// <summary>
    /// Clears the previous run snapshot so every run body starts without
    /// stale diagnostics, failures, or profiles.
    /// </summary>
    private void ResetRunDiagnostics()
    {
        LastErrorMessage = null;
        LastFailedNodeName = null;
        LastFailedNodeOp = null;
        LastErrorCause = null;
        LastProfile = null;
        LastRunTime = TimeSpan.Zero;
        LastAllocatedBytes = 0;
        LastScratchBytes = 0;
        LastCopyBytes = 0;
        LastPeakLiveBytes = 0;
        LastGcCollections = new int[3];
        liveArrayUsers = null;
        liveUnknownTensors = null;
        liveIndexSeeded = false;
    }

    /// <summary>
    /// Validates the execution options at run entry, recording the cause
    /// naming the defect like every other entry rejection.
    /// </summary>
    private bool ValidateRunOptions()
    {
        try
        {
            (Options ?? ExecutionOptions.Default).Validated();
        }
        catch (Exception ex)
        {
            return Fail("Invalid execution options: {m}.", ex.Message);
        }
        return true;
    }

    /// <summary>
    /// Records a binding or lifecycle rejection: logs it, snapshots it as the
    /// current error with no node identity, drops run outputs and returns false.
    /// </summary>
    protected bool Fail(string messageTemplate, params object?[] args)
    {
        Error(messageTemplate, args);
        LastErrorMessage = Log.Render(messageTemplate, args);
        LastFailedNodeName = null;
        LastFailedNodeOp = null;
        LastErrorCause = null;
        InvalidateOutputs();
        return false;
    }

    /// <summary>
    /// Records a node failure: logs it with its original exception when one was
    /// captured, snapshots message, node identity and cause, drops run outputs
    /// and returns false.
    /// </summary>
    protected bool FailNode(Node node, Exception? cause, string messageTemplate, params object?[] args)
    {
        if (cause is not null) Error(cause, messageTemplate, args);
        else Error(messageTemplate, args);
        LastErrorMessage = Log.Render(messageTemplate, args);
        LastFailedNodeName = node.Name;
        LastFailedNodeOp = node.Op;
        LastErrorCause = cause;
        InvalidateOutputs();
        return false;
    }

    public void InvalidateOutputs()
    {
        Outputs.Clear();
        foreach (var key in IntermediateOutputs.Keys.ToArray())
        {
            IntermediateOutputs[key] = null;
        }
    }

    public bool ExecuteNode(object userInputs, string nodeLabel, bool useInitializers) => ExecuteNode(userInputs, nodeLabel, useInitializers, ExecutionProvider.CPU, null);

    /// <summary>
    /// Runs only the single node named <paramref name="nodeLabel"/> with explicitly
    /// supplied inputs; upstream nodes do not run, so every node input must come
    /// from the caller or from initializers. On success the graph outputs hold
    /// exactly this node outputs. An unknown label fails like any other execution error.
    /// </summary>
    /// <param name="userInputs">Caller-owned input tensors; the graph never takes ownership of their storage.</param>
    /// <param name="nodeLabel">Name of the node to run.</param>
    /// <param name="useInitializers">Bind stored initializers for graph inputs left unspecified.</param>
    /// <param name="provider">Execution provider carrying out the operators.</param>
    /// <param name="options">Execution options, or null for the graph prepared options.</param>
    /// <returns>True on success; otherwise false with details on LastErrorMessage, LastFailedNodeName, LastFailedNodeOp and LastErrorCause.</returns>
    public virtual bool ExecuteNode(object userInputs, string nodeLabel, bool useInitializers, ExecutionProvider provider, ExecutionOptions? options)
    {
        if (Interlocked.CompareExchange(ref _executing, 1, 0) != 0)
        {
            return false;
        }
        try
        {
            GraphExecution exec;
            lock (PrepareLock)
            {
                EnsurePreparedLocked();
                exec = new GraphExecution(this, options, _prepared, _preparedFingerprint, _preparationError);
            }
            bool ok = exec.RunNodeCore(userInputs, nodeLabel, useInitializers, provider);
            CopyFromExecution(exec);
            return ok;
        }
        finally { Volatile.Write(ref _executing, 0); }
    }

    protected bool RunNodeCore(object userInputs, string nodeLabel, bool useInitializers, ExecutionProvider provider)
    {
        ResetRunDiagnostics();
        long start = Stopwatch.GetTimestamp();
        long allocStart = GC.GetAllocatedBytesForCurrentThread();
        int gc0 = GC.CollectionCount(0);
        int gc1 = GC.CollectionCount(1);
        int gc2 = GC.CollectionCount(2);
        try
        {
            return RunNodeCoreInner(userInputs, nodeLabel, useInitializers, provider);
        }
        finally
        {
            LastRunTime = Stopwatch.GetElapsedTime(start);
            LastAllocatedBytes = GC.GetAllocatedBytesForCurrentThread() - allocStart;
            LastGcCollections = new int[] { GC.CollectionCount(0) - gc0, GC.CollectionCount(1) - gc1, GC.CollectionCount(2) - gc2 };
        }
    }

    bool RunNodeCoreInner(object userInputs, string nodeLabel, bool useInitializers, ExecutionProvider provider)
    {
        if (!ValidateRunOptions()) return false;
        // Node is a struct, so a miss yields default(Node) with null Name;
        // search by index instead of comparing against an empty name.
        var nodeIndex = Nodes.FindIndex(n => n.Name == nodeLabel);
        if (nodeIndex < 0)
        {
            return Fail("Could not find node {n} in graph.", nodeLabel);
        }
        var node = Nodes[nodeIndex];
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
            return Fail("Unsupported user inputs type: {t}.", userInputs.GetType().Name);
        }

        using var op = Begin("Executing node {node} in graph {n} from {f}", nodeLabel, Metadata["Name"], ModelFile);
        if (Log.IsEnabled(LogLevel.Debug))
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
        }
        OpResult r;
        try
        {
            r = node.Execute(this, provider, Options);
        }
        catch (Exception ex)
        {
            FailNode(node, ex, "Execution of node {n} with op {op} threw {t}: {m}.", node.Name, node.Op, ex.GetType().Name, ex.Message);
            throw;
        }
        if (r.Status == OpStatus.Failure)
        {
            FailNode(node, r.Cause, "Execution of node {n} with op {op} failed: {m}.", node.Name, node.Op, r.Message ?? "");
            return false;
        }
        else
        {
            Debug("Execution of node {n} with op {op} returned {s} with {c} output(s).", node.Name, node.Op.ToString(), r.Status.ToString(), r.Outputs.Length);
            foreach (var kv in Outputs) RemoveLiveRefs(kv.Value);
            Outputs.Clear();
            for (int i = 0; i < node.Outputs.Length; i++)
            {
                if (string.IsNullOrEmpty(node.Outputs[i])) continue;
                Debug("Assigning node {n} output {c} to graph tensor {o}.", node.Name, i, node.Outputs[i]);
                TrackBind(Outputs, node.Outputs[i], r.Outputs[i]);
            }
        }
        op.Complete();
        return true;
    }

    public void Reset() => Reset(false);

    public void Reset(bool gc)
    {
        foreach (var o in IntermediateOutputs.Keys)
        {
            IntermediateOutputs[o] = null;
        }
        Outputs.Clear();
        foreach (var vp in OutputDescs)
        {
            if (string.IsNullOrEmpty(vp.Name)) continue;
            Outputs.MarkUnresolved(vp.Name);
        }
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
        // Preparation assigns stable sequential identities by file-order
        // position: unlike name hashes they are distinct for duplicate or
        // anonymous names and identical across processes.
        for (int i = 0; i < Nodes.Count; i++)
        {
            var node = Nodes[i];
            node.ID = i;
            NormalizeIntArrayAttributes(node.Attributes);
            Nodes[i] = node;
        }
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
        _prepared = true;
        _preparedFingerprint = ComputeStructureFingerprint();
        _preparationError = ValidatePreparation();
    }

    /// <summary>
    /// Replaces lossless long integer-array attributes with integer arrays
    /// so steady-state runs read the stored values without converting.
    /// Values that would overflow stay untouched, keeping their run-time
    /// failure behavior. Imported graphs already arrive canonicalized.
    /// </summary>
    static void NormalizeIntArrayAttributes(Dictionary<string, object>? attributes)
    {
        if (attributes is null) return;
        foreach (var kv in attributes)
        {
            if (kv.Value is long[])
            {
                ConvertLongArrayAttributes(attributes);
                return;
            }
        }
    }

    static void ConvertLongArrayAttributes(Dictionary<string, object> attributes)
    {
        foreach (var key in attributes.Keys.ToArray())
        {
            if (attributes[key] is long[] source && IsLosslessIntRange(source))
            {
                var converted = new int[source.Length];
                for (int i = 0; i < source.Length; i++) converted[i] = (int)source[i];
                attributes[key] = converted;
            }
        }
    }

    static bool IsLosslessIntRange(long[] source)
    {
        for (int i = 0; i < source.Length; i++)
        {
            if (source[i] < int.MinValue || source[i] > int.MaxValue) return false;
        }
        return true;
    }

    static string NodeLabel(Node node, int index) =>
        string.IsNullOrEmpty(node.Name) ? "#" + index : node.Name;

    long ComputeStructureFingerprint()
    {
        unchecked
        {
            ulong h = 1469598103934665603UL;
            void MixUlong(ulong v)
            {
                h ^= v;
                h *= 1099511628211UL;
            }
            void MixInt(int v) => MixUlong((ulong)(uint)v);
            void MixString(string? v)
            {
                if (v is null)
                {
                    MixUlong(0x9E3779B97F4A7C15UL);
                    return;
                }
                MixInt(v.Length);
                foreach (char c in v) MixUlong((ulong)c);
            }
            MixInt(Nodes.Count);
            for (int i = 0; i < Nodes.Count; i++)
            {
                var node = Nodes[i];
                MixString(node.Name);
                MixInt((int)node.Op);
                MixString(node.Domain);
                MixString(node.OpTypeName);
                if (node.Inputs is null) MixInt(-1);
                else
                {
                    MixInt(node.Inputs.Length);
                    foreach (var input in node.Inputs) MixString(input);
                }
                if (node.Outputs is null) MixInt(-1);
                else
                {
                    MixInt(node.Outputs.Length);
                    foreach (var output in node.Outputs) MixString(output);
                }
            }
            MixInt(Inputs.Count);
            foreach (var key in Inputs.Keys) MixString(key);
            MixInt(Initializers.Count);
            foreach (var key in Initializers.Keys) MixString(key);
            MixInt(InputDescs.Count);
            foreach (var desc in InputDescs) MixString(desc?.Name);
            MixInt(OutputDescs.Count);
            foreach (var desc in OutputDescs) MixString(desc?.Name);
            return (long)h;
        }
    }

    string? ValidatePreparation()
    {
        var producer = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < Nodes.Count; i++)
        {
            var outputs = Nodes[i].Outputs;
            if (outputs is null) continue;
            foreach (var output in outputs)
            {
                if (string.IsNullOrEmpty(output)) continue;
                if (producer.TryGetValue(output, out var first))
                {
                    return "Tensor " + output + " has duplicate producers "
                        + NodeLabel(Nodes[first], first) + " and " + NodeLabel(Nodes[i], i) + ".";
                }
                producer[output] = i;
            }
        }
        for (int i = 0; i < Nodes.Count; i++)
        {
            var inputs = Nodes[i].Inputs;
            if (inputs is null) continue;
            foreach (var input in inputs)
            {
                if (string.IsNullOrEmpty(input)) continue;
                if (producer.TryGetValue(input, out var pi) && pi >= i)
                {
                    return "Tensor " + input + " is consumed by node " + NodeLabel(Nodes[i], i)
                        + " before its producer " + NodeLabel(Nodes[pi], pi) + ".";
                }
            }
        }
        foreach (var desc in OutputDescs)
        {
            if (desc is null || string.IsNullOrEmpty(desc.Name)) continue;
            if (producer.ContainsKey(desc.Name)) continue;
            if (Inputs.ContainsKey(desc.Name)) continue;
            if (Initializers.ContainsKey(desc.Name)) continue;
            return "Graph output " + desc.Name + " has no producer, input, or initializer.";
        }
        return null;
    }

    readonly struct ExecutionPoolScope : IDisposable
    {
        readonly ComputationalGraph graph;
        public ExecutionPoolScope(ComputationalGraph graph)
        {
            this.graph = graph;
            graph.ActivePool = new TensorBufferPool();
            graph.ActiveScratch = new ScratchAccountant();
            graph.ActiveCopy = new CopyAccountant();
        }
        public void Dispose()
        {
            if (graph.ActivePool is not null)
            {
                graph.LastPoolAllocatedNew = graph.ActivePool.AllocatedNew;
                graph.LastPoolReused = graph.ActivePool.Reused;
                graph.LastPoolReturned = graph.ActivePool.Returned;
                graph.LastPoolDropped = graph.ActivePool.Dropped;
                graph.LastPoolAllocatedNewBytes = graph.ActivePool.AllocatedNewBytes;
                graph.LastPoolReusedBytes = graph.ActivePool.ReusedBytes;
                graph.LastPoolPeakOutstandingBytes = graph.ActivePool.PeakOutstandingBytes;
            }
            if (graph.ActiveScratch is not null)
            {
                graph.LastScratchBytes = graph.ActiveScratch.TotalScratchBytes;
            }
            if (graph.ActiveCopy is not null)
            {
                graph.LastCopyBytes = graph.ActiveCopy.TotalCopyBytes;
            }
            graph.ActiveScratch = null;
            graph.ActiveCopy = null;
            graph.ActivePool = null;
        }
    }

    /// <summary>
    /// Drops one dead reference, returning exactly-backed pool-owned storage
    /// with no live aliases. Reference dropping covers every dtype and layout
    /// (letting the collector reclaim); storage return additionally requires
    /// pool ownership. Owned storage pinned by a live view stays readable (its
    /// observer may still hold the alias); graph outputs are never touched.
    /// </summary>
    /// <returns>True when the reference was dropped.</returns>
    bool TryReleaseValue(string name, ref HashSet<Array>? returned)
    {
        var pool = ActivePool;
        if (pool is null) return false;
        if (!IntermediateOutputs.TryGetValue(name, out var tensor) || tensor is null) return false;
        if (Outputs.ContainsKey(name)) return false;
        var arr = (tensor as TensorBase)?.OwnedBufferArray();
        if (arr is not null && pool.IsOwned(arr) && !(returned?.Contains(arr) ?? false))
        {
            EnsureLiveIndexSeeded();
            RemoveLiveRefs(tensor);
            if (!HasLiveAliasIndexed(arr, pool))
            {
                pool.Return(arr);
                returned ??= new HashSet<Array>();
                returned.Add(arr);
                IntermediateOutputs[name] = null;
                return true;
            }
            AddLiveRefs(tensor);
            return false;
        }
        RemoveLiveRefs(tensor);
        IntermediateOutputs[name] = null;
        return true;
    }

    void ReleaseDeadTensors(Node node, int index, ref List<string>? pending)
    {
        var pool = ActivePool;
        if (pool is null || node.Inputs is null) return;
        HashSet<Array>? returned = null;
        foreach (var name in node.Inputs)
        {
            if (string.IsNullOrEmpty(name)) continue;
            if (!LastUseIndex.TryGetValue(name, out var last) || last != index) continue;
            if (node.Outputs is not null && Array.IndexOf(node.Outputs, name) >= 0)
            {
                DeferRelease(name, ref pending);
                continue;
            }
            if (TryReleaseValue(name, ref returned) == false) DeferRelease(name, ref pending);
        }
        // Reclaiming after the final node serves no later rent, so only
        // earlier nodes retry values that died behind a live alias plus
        // outputs this node produced but nobody consumes.
        if (index >= Nodes.Count - 1) return;
        if (node.Outputs is not null)
        {
            foreach (var name in node.Outputs)
            {
                if (string.IsNullOrEmpty(name)) continue;
                if (!LastUseIndex.TryGetValue(name, out var last) || last != index) continue;
                if (TryReleaseValue(name, ref returned) == false) DeferRelease(name, ref pending);
            }
        }
        RetryPendingReleases(ref pending, ref returned);
    }

    /// <summary>
    /// Parks a still-live intermediate for a later retry instead of rescanning
    /// every intermediate after every node. Released slots and graph outputs
    /// never park: the former stay dead, the latter stay live to the end.
    /// </summary>
    void DeferRelease(string name, ref List<string>? pending)
    {
        if (Outputs.ContainsKey(name)) return;
        if (!IntermediateOutputs.TryGetValue(name, out var tensor) || tensor is null) return;
        pending ??= new List<string>();
        if (pending.Contains(name) == false) pending.Add(name);
    }

    /// <summary>
    /// Retries intermediates whose storage stayed pinned by a live alias when
    /// they died. Reclaimed names leave the pending set; names that became
    /// graph outputs drop out without touching their bindings.
    /// </summary>
    void RetryPendingReleases(ref List<string>? pending, ref HashSet<Array>? returned)
    {
        if (pending is null || pending.Count == 0) return;
        for (int i = pending.Count - 1; i >= 0; i--)
        {
            var name = pending[i];
            if (Outputs.ContainsKey(name))
            {
                pending.RemoveAt(i);
            }
            else if (TryReleaseValue(name, ref returned))
            {
                pending.RemoveAt(i);
            }
        }
    }

    /// <summary>
    /// Whether any live binding besides the already-uncounted dying tensor
    /// shares the candidate array. Static bindings use the memoized roots;
    /// the variable maps use the reverse index, falling back to conservative
    /// true while unknown-kind tensors are live.
    /// </summary>
    static int ElementByteSize(TensorElementType element) => element switch
    {
        TensorElementType.Bool or TensorElementType.Int8 or TensorElementType.UInt8 => 1,
        TensorElementType.Int16 or TensorElementType.UInt16 or TensorElementType.Float16 or TensorElementType.BFloat16 => 2,
        TensorElementType.Int32 or TensorElementType.UInt32 or TensorElementType.Float => 4,
        TensorElementType.Int64 or TensorElementType.UInt64 or TensorElementType.Double or TensorElementType.Complex64 => 8,
        TensorElementType.Complex128 => 16,
        _ => 0,
    };

    /// <summary>Sums logical payload bytes over the run bindings (inputs, outputs, live intermediates).</summary>
    /// <remarks>Counts per binding without deduplicating aliased storage, matching the manual
    /// live-at-end census; initializers are static model data, not run pressure, and stay out.
    /// Allocation-free: struct enumerators plus integer arithmetic only.</remarks>
    long LivePayloadBytes()
    {
        long total = 0;
        foreach (var kv in Inputs) if (kv.Value is ITensor t) total += t.Length * ElementByteSize(t.ElementType);
        foreach (var kv in Outputs) if (kv.Value is ITensor t) total += t.Length * ElementByteSize(t.ElementType);
        foreach (var kv in IntermediateOutputs) if (kv.Value is ITensor t) total += t.Length * ElementByteSize(t.ElementType);
        return total;
    }

    void NoteLivePeak()
    {
        long live = LivePayloadBytes();
        if (live > LastPeakLiveBytes) LastPeakLiveBytes = live;
    }

    bool HasLiveAliasIndexed(Array candidate, TensorBufferPool pool)
    {
        var staticRoots = EnsurePoolRoots(pool);
        if (staticRoots is not null)
        {
            if (staticRoots.Contains(candidate)) return true;
        }
        else
        {
            foreach (var tensor in Inputs.Values) if (SharesPooledStorage(candidate, tensor)) return true;
            foreach (var tensor in Initializers.Values) if (SharesPooledStorage(candidate, tensor)) return true;
            foreach (var attr in EnumerateAttributeTensors()) if (SharesPooledStorage(candidate, attr)) return true;
        }
        if (liveUnknownTensors is not null && liveUnknownTensors.Count > 0) return true;
        return liveArrayUsers is not null && liveArrayUsers.TryGetValue(candidate, out int users) && users > 0;
    }

    void EnsureLiveIndexSeeded()
    {
        if (liveIndexSeeded) return;
        liveIndexSeeded = true;
        liveArrayUsers = new Dictionary<Array, int>();
        liveUnknownTensors = new HashSet<ITensor>(ReferenceEqualityComparer.Instance);
        foreach (var kv in IntermediateOutputs) AddLiveRefs(kv.Value);
        foreach (var kv in Outputs) AddLiveRefs(kv.Value);
    }

    void TrackBind(IDictionary<string, ITensor?> map, string name, ITensor? value)
    {
        if (map.TryGetValue(name, out var old) && !ReferenceEquals(old, value)) RemoveLiveRefs(old);
        map[name] = value;
        AddLiveRefs(value);
    }

    void AddLiveRefs(ITensor? tensor)
    {
        if (tensor is null || liveArrayUsers is null || liveUnknownTensors is null) return;
        var roots = new HashSet<Array>();
        if (CollectAliasRoot(tensor, roots))
        {
            foreach (var r in roots) liveArrayUsers[r] = liveArrayUsers.TryGetValue(r, out int c) ? c + 1 : 1;
        }
        else liveUnknownTensors.Add(tensor);
    }

    void RemoveLiveRefs(ITensor? tensor)
    {
        if (tensor is null || liveArrayUsers is null || liveUnknownTensors is null) return;
        if (liveUnknownTensors.Remove(tensor)) return;
        var roots = new HashSet<Array>();
        if (CollectAliasRoot(tensor, roots))
        {
            foreach (var r in roots)
            {
                if (liveArrayUsers.TryGetValue(r, out int c)) liveArrayUsers[r] = c - 1;
            }
        }
    }

    /// <summary>
    /// Memoizes the run-static alias snapshot on the per-execution pool so
    /// graphs without pool-owned releases never pay for it. A null snapshot
    /// selects the legacy per-release scan for every probe of the run.
    /// </summary>
    HashSet<Array>? EnsurePoolRoots(TensorBufferPool pool)
    {
        if (pool.StaticRootsBuilt) return pool.StaticRoots;
        pool.StaticRootsBuilt = true;
        pool.StaticRoots = BuildStaticAliasRoots();
        return pool.StaticRoots;
    }

    /// <summary>
    /// Snapshots the backing arrays of run-static tensors (bound inputs,
    /// initializers and attribute tensors) so alias probes do one set lookup
    /// instead of rescanning constants per release. Kernels only read these
    /// bindings during a run, so the snapshot stays valid; anything that
    /// cannot be reduced to array roots selects the legacy scan by
    /// returning null. The structure fingerprint ignores tensor values, so
    /// this snapshot is per-run rather than per-preparation.
    /// </summary>
    HashSet<Array>? BuildStaticAliasRoots()
    {
        var roots = new HashSet<Array>(Inputs.Count + Initializers.Count + 2 * Nodes.Count);
        foreach (var tensor in Inputs.Values)
        {
            if (CollectAliasRoot(tensor, roots) == false) return null;
        }
        foreach (var tensor in Initializers.Values)
        {
            if (CollectAliasRoot(tensor, roots) == false) return null;
        }
        foreach (var attr in EnumerateAttributeTensors())
        {
            if (CollectAliasRoot(attr, roots) == false) return null;
        }
        return roots;
    }

    /// <summary>
    /// Adds the ultimate dense backing arrays of one static tensor, mirroring
    /// SharesPooledStorage traversal. Returns false for non-array dense
    /// buffers and unknown view kinds, keeping the probe conservative.
    /// </summary>
    static bool CollectAliasRoot(ITensor? tensor, HashSet<Array> roots)
    {
        if (tensor is null) return true;
        if (tensor is TensorSequence sequence)
        {
            foreach (var item in sequence.Items)
            {
                if (CollectAliasRoot(item, roots) == false) return false;
            }
            return true;
        }
        if (tensor is not Tensor<float> typed) return true;
        if (typed is DenseTensor<float> dense)
        {
            if (MemoryMarshal.TryGetArray(dense.Buffer, out ArraySegment<float> segment) && segment.Array is not null)
            {
                roots.Add(segment.Array);
                return true;
            }
            return false;
        }
        if (typed is BroadcastedTensor<float> broadcast) return CollectAliasRoot(broadcast.source, roots);
        if (typed is TensorSlice<float> slice) return CollectAliasRoot(slice.parent, roots);
        return false;
    }

    IEnumerable<ITensor> EnumerateAttributeTensors()
    {
        foreach (var n in Nodes)
        {
            var attrs = n.Attributes;
            if (attrs is null) continue;
            foreach (var value in attrs.Values)
            {
                if (value is ITensor single)
                {
                    yield return single;
                }
                else if (value is System.Collections.IEnumerable enumerable && value is not string)
                {
                    foreach (var item in enumerable)
                    {
                        if (item is ITensor inner) yield return inner;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Checks one runtime value against a pooled backing array with explicit
    /// per-kind storage traversal: sequences recurse into elements, views
    /// recurse into their sources and parents, and dense tensors check their
    /// value buffers. Unknown future tensor kinds stay conservative
    /// (true) so arrays are never returned while possibly referenced. The pool
    /// manages float backing arrays, so non-float tensors cannot alias.
    /// </summary>
    static bool SharesPooledStorage(Array candidate, ITensor? tensor)
    {
        if (tensor is null) return false;
        if (tensor is TensorSequence seq)
        {
            foreach (var item in seq.Items)
            {
                if (SharesPooledStorage(candidate, item)) return true;
            }
            return false;
        }
        if (tensor is not Tensor<float> typed) return false;
        if (typed is DenseTensor<float> dense) return SharesBuffer(candidate, dense.Buffer);
        if (typed is BroadcastedTensor<float> broadcast) return SharesPooledStorage(candidate, broadcast.source);
        if (typed is TensorSlice<float> slice) return SharesPooledStorage(candidate, slice.parent);
        return true;
    }

    static bool SharesBuffer(Array candidate, Memory<float> buffer)
    {
        if (MemoryMarshal.TryGetArray(buffer, out ArraySegment<float> segment)) return ReferenceEquals(segment.Array, candidate);
        return true;
    }
    #endregion
}

