namespace Lokad.Onnx;

public partial class ComputationalGraph
{
    internal Dictionary<long, string[]>? nodeReadNames;
    internal IReadOnlyDictionary<string, ITensor>? CapturedInputs;
    List<SubgraphExecutionInfo>? subgraphExecutions;

    /// <summary>Separate diagnostics for nested executions, keyed by their owning node and branch.</summary>
    /// <remarks>The parent's wall time and GC allocation are inclusive. Its pool, scratch,
    /// copy and live-payload counters describe its own scope; each child records its own
    /// counters here. Independent pool peaks must not be summed as a process-memory peak.
    /// Child node identifiers belong to the child scope, not to the parent's node list.</remarks>
    public IReadOnlyList<SubgraphExecutionInfo> LastSubgraphExecutions =>
        subgraphExecutions is null ? Array.Empty<SubgraphExecutionInfo>() : subgraphExecutions.AsReadOnly();

    string[] NodeInputs(Node node) => nodeReadNames is not null && nodeReadNames.TryGetValue(node.ID, out var reads)
        ? reads : node.Inputs ?? Array.Empty<string>();

    internal void BindCaptures(IReadOnlyDictionary<string, ITensor> captures)
    {
        CapturedInputs = captures;
        // Captures are roots in this execution only. They are not formal graph
        // inputs and never alter the shared plan's dictionaries or descriptors.
        _preparationError = ValidatePreparation();
    }

    internal void RecordSubgraph(Node node, string branch, GraphExecution execution, bool succeeded)
    {
        subgraphExecutions ??= new List<SubgraphExecutionInfo>();
        subgraphExecutions.Add(new SubgraphExecutionInfo(node, branch, execution, succeeded));
    }
}

/// <summary>A completed nested execution's diagnostics, without retaining its tensors or context.</summary>
public sealed class SubgraphExecutionInfo
{
    public long NodeId { get; }
    public string NodeName { get; }
    public string Branch { get; }
    public bool Succeeded { get; }
    public string? Error { get; }
    public TimeSpan RunTime { get; }
    public long AllocatedBytes { get; }
    public long PoolAllocatedNewBytes { get; }
    public long PoolReusedBytes { get; }
    public long PoolPeakOutstandingBytes { get; }
    public long ScratchBytes { get; }
    public long CopyBytes { get; }
    public long PeakLiveBytes { get; }
    public IReadOnlyList<WallNode> WallProfile { get; }
    public IReadOnlyList<NodeProfile> Profile { get; }
    public IReadOnlyList<SubgraphExecutionInfo> Children { get; }

    internal SubgraphExecutionInfo(Node node, string branch, GraphExecution execution, bool succeeded)
    {
        NodeId = node.ID; NodeName = node.Name; Branch = branch; Succeeded = succeeded;
        Error = execution.LastErrorMessage; RunTime = execution.LastRunTime;
        AllocatedBytes = execution.LastAllocatedBytes;
        PoolAllocatedNewBytes = execution.LastPoolAllocatedNewBytes;
        PoolReusedBytes = execution.LastPoolReusedBytes;
        PoolPeakOutstandingBytes = execution.LastPoolPeakOutstandingBytes;
        ScratchBytes = execution.LastScratchBytes; CopyBytes = execution.LastCopyBytes;
        PeakLiveBytes = execution.LastPeakLiveBytes;
        WallProfile = Array.AsReadOnly(execution.LastWallProfile?.ToArray() ?? Array.Empty<WallNode>());
        Profile = Array.AsReadOnly(execution.LastProfile?.ToArray() ?? Array.Empty<NodeProfile>());
        Children = execution.LastSubgraphExecutions;
    }
}
