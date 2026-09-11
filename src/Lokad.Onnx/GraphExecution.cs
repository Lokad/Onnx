namespace Lokad.Onnx;

using static Lokad.Onnx.Runtime;

/// <summary>
/// An isolated execution context for one <see cref="ComputationalGraph"/> run.
/// </summary>
/// <remarks>
/// A context shares the prepared plan (nodes, initializers, descriptors,
/// opsets, lifetime analysis) by reference and owns all run state: input,
/// output and intermediate bindings, the buffer pool, and error/profile
/// reporting. Separate contexts over one graph execute concurrently without
/// sharing mutable state; sequential <see cref="ComputationalGraph"/> calls
/// delegate to a fresh context per run and copy results back.
/// </remarks>
public sealed class GraphExecution : ComputationalGraph
{
    internal GraphExecution(ComputationalGraph prepared, ExecutionOptions? options, bool preparedFlag, long preparedFingerprint, string? preparationError)
    {
        _prepared = preparedFlag;
        _preparedFingerprint = preparedFingerprint;
        _preparationError = preparationError;
        Nodes = prepared.Nodes;
        Initializers = prepared.Initializers;
        FoldedTransposes = prepared.FoldedTransposes;
        PackedWeights = prepared.PackedWeights;
        FoldLock = prepared.FoldLock;
        PrepareLock = prepared.PrepareLock;
        InputDescs = prepared.InputDescs;
        OutputDescs = prepared.OutputDescs;
        Opset = prepared.Opset;
        Metadata = prepared.Metadata;
        MetadataProps = prepared.MetadataProps;
        Options = options ?? prepared.Options;
        LastUseIndex = prepared.LastUseIndex;
        ModelFile = prepared.ModelFile;
        Inputs = new BindingMap(prepared.Inputs);
        Outputs = new BindingMap(prepared.Outputs);
        IntermediateOutputs = new Dictionary<string, ITensor?>(prepared.IntermediateOutputs);
    }

    public override bool Execute(object userInputs, bool useInitializers, ExecutionProvider provider, ExecutionOptions? options)
    {
        if (System.Threading.Interlocked.CompareExchange(ref _executing, 1, 0) != 0)
        {
            return false;
        }
        try
        {
            if (options is not null) Options = options;
            EnsurePrepared();
            return RunCore(userInputs, useInitializers, provider);
        }
        finally { System.Threading.Volatile.Write(ref _executing, 0); }
    }

    public override bool ExecuteNode(object userInputs, string nodeLabel, bool useInitializers, ExecutionProvider provider, ExecutionOptions? options)
    {
        if (System.Threading.Interlocked.CompareExchange(ref _executing, 1, 0) != 0)
        {
            return false;
        }
        try
        {
            if (options is not null) Options = options;
            EnsurePrepared();
            return RunNodeCore(userInputs, nodeLabel, useInitializers, provider);
        }
        finally { System.Threading.Volatile.Write(ref _executing, 0); }
    }
}
