namespace Lokad.Onnx;

/// <summary>One step of a blocked residual region: a Conv, plain Relu, or residual Add over blocked buffers.</summary>
internal enum BlockedRegionStepKind
{
    Conv,
    Add,
    Relu,
}

/// <summary>A single fused step. DataInput is the chain value (Conv data, Relu input, Add leg); AuxInput is the Add second leg; FilterName and BiasName serve Conv (bias null means none); ReluAfter marks a fused Conv epilogue. Output names the value the step binds.</summary>
internal sealed record BlockedRegionStep(
    BlockedRegionStepKind Kind,
    string Output,
    string DataInput,
    string? AuxInput,
    string? FilterName,
    string? BiasName,
    bool ReluAfter);

/// <summary>A maximal blocked-execution region: convert the input once, run every step on blocked buffers with prepared filters, convert the output once.</summary>
internal sealed record BlockedRegionSpec(
    string InputName,
    BlockedRegionStep[] Steps,
    string OutputName);

/// <summary>A filter packed into the blocked micro-tile layout, with the source identity it was built from for per-run validation.</summary>
internal sealed record PreparedBlockedFilter(
    float[] Packed,
    ITensor SourceRef,
    long SourceLength,
    int M,
    int C);

