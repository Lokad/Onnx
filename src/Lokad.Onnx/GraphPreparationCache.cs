namespace Lokad.Onnx;

/// <summary>
/// Prepared weights and synchronization shared by a graph and its execution
/// contexts. Bindings, lifetime fingerprints and run diagnostics stay on each
/// instance; public graph mutations still use the existing invalidation rules.
/// </summary>
internal sealed class GraphPreparationCache
{
    internal readonly Dictionary<string, ComputationalGraph.FoldedTranspose> FoldedTransposes = new(StringComparer.Ordinal);
    internal readonly Dictionary<float[], PackedMatMulWeight> PackedWeights = new();
    internal readonly Dictionary<float[], PackedConvWeight> PackedConvWeights = new();
    internal readonly Dictionary<float[], PackedLstmWeight> PackedLstmWeights = new();
    internal readonly object FoldLock = new();
    internal readonly object PrepareLock = new();
    internal bool Winograd4;
}
