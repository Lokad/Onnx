namespace Lokad.Onnx;

using System.Runtime.Intrinsics.X86;

public partial class ComputationalGraph
{
    internal int OwnedPackedWeightCount
    {
        get { int count = 0; foreach (var value in Initializers.Values) if (value is OwnedPackedTensor) count++; return count; }
    }
    internal long OwnedPackedWeightBytes
    {
        get
        {
            long bytes = 0; var seen = new HashSet<float[]>();
            foreach (var value in Initializers.Values)
                if (value is OwnedPackedTensor packed && seen.Add(packed.PackedArray)) bytes += (long)packed.PackedArray.Length * sizeof(float);
            return bytes;
        }
    }

    static Dictionary<string, (int Count, string? ConsumerName, bool Pointwise)> OwnedWeightUses(IReadOnlyList<Node> nodes)
    {
        var uses = new Dictionary<string, (int Count, string? ConsumerName, bool Pointwise)>(StringComparer.Ordinal);
        foreach (var node in nodes)
        {
            var inputs = GraphCaptures.NodeInputs(node);
            for (int i = 0; i < inputs.Length; i++)
            {
                string name = inputs[i]; if (string.IsNullOrEmpty(name)) continue;
                bool pointwise = node.Op == OpType.FusedConformerConv && Node.IsStandardDomain(node.Domain)
                    && inputs.Length == 8 && i is 2 or 5;
                bool eligible = pointwise || node.Op == OpType.MatMul && Node.IsStandardDomain(node.Domain) && i == 1 && inputs.Length == 2;
                uses.TryGetValue(name, out var prior);
                uses[name] = (prior.Count + 1, eligible && prior.Count == 0 ? node.Name : null, pointwise && prior.Count == 0);
            }
        }
        return uses;
    }

    /// <summary>Replaces eligible float matrix initializers with independently owned packed storage.</summary>
    /// <returns>The number of initializers replaced; zero when no eligible weight or hardware is available.</returns>
    /// <remarks>
    /// Explicit opt-in for the supported Parakeet feed-forward, square attention and fused pointwise-region weights.
    /// Call while holding exclusive access to a newly loaded graph, before creating execution contexts.
    /// AVX512 hosts also fuse qualified pointwise regions when both weights can be owned.
    /// Ordinary Prepare calls never opt in. Logical values and previously held source arrays are
    /// preserved; the graph replaces each eligible initializer and can release its old reference.
    /// Shared, visible, captured and already prepared weights remain untouched. Repeated calls are
    /// idempotent. This replaces initializer storage rather than adding a retained weight-cache entry.
    /// If a later allocation fails, completed replacements leave the graph executable.
    /// </remarks>
    /// <exception cref="InvalidOperationException">The graph is executing when preparation is attempted.</exception>
    public int PrepareOwnedMatMulWeights()
    {
        if (!Fma.IsSupported) return 0;
        if (System.Threading.Volatile.Read(ref _executing) != 0)
            throw new InvalidOperationException("Owned preparation is not allowed during execution.");
        lock (PrepareLock)
        {
            if (System.Threading.Volatile.Read(ref _executing) != 0)
                throw new InvalidOperationException("Owned preparation is not allowed during execution.");
            // A direct consumer census is insufficient when branches can capture names.
            if (CapturedInputs is { Count: > 0 }) return 0;
            foreach (var node in Nodes)
                if (node.Attributes is not null)
                    foreach (var value in node.Attributes.Values) if (value is ComputationalGraph) return 0;
            EnsurePreparedLocked();
            var uses = OwnedWeightUses(Nodes);
            var protectedRoots = new HashSet<Array>();
            foreach (var value in Inputs.Values.Concat(Outputs.Values).Concat(EnumerateAttributeTensors()))
                if (!CollectAliasRoot(value, protectedRoots)) return 0;
            foreach (var entry in PackedWeights)
            {
                protectedRoots.Add(entry.Key);
                if (!CollectAliasRoot(entry.Value.Packed, protectedRoots)) return 0;
            }
            foreach (var entry in PackedConvWeights)
            {
                protectedRoots.Add(entry.Key); protectedRoots.Add(entry.Value.Values);
                if (entry.Value.WinogradValues is { } winograd) protectedRoots.Add(winograd);
            }
            foreach (var entry in PackedLstmWeights)
            { protectedRoots.Add(entry.Key); protectedRoots.Add(entry.Value.Values); }
            var folded = new HashSet<string>(StringComparer.Ordinal);
            foreach (var entry in FoldedTransposes.Values)
            { folded.Add(entry.SourceName); folded.Add(entry.PreparedName); }
            var bindings = new Dictionary<Array, int>();
            var roots = new HashSet<Array>();
            foreach (var value in Initializers.Values)
            {
                roots.Clear();
                if (!CollectAliasRoot(value, roots)) return 0;
                foreach (var root in roots) bindings[root] = bindings.TryGetValue(root, out int count) ? count + 1 : 1;
            }
            roots.Clear();
            bool Candidate(string name, string consumer, bool pointwise, out DenseTensor<float> source, out float[] array, out int n, out int k)
            {
                source = null!; array = null!; n = k = 0;
                if (!uses.TryGetValue(name, out var use) || use.Count != 1
                    || Inputs.ContainsKey(name) || Outputs.ContainsKey(name) || folded.Contains(name)
                    || Initializers[name] is not DenseTensor<float> dense || !GraphConvPacking.Standard(dense)) return false;
                source = dense;
                if (pointwise)
                {
                    if (source.Rank != 3 || source.Dimensions[2] != 1) return false;
                    n = source.Dimensions[1]; k = source.Dimensions[0];
                    if (!ParakeetOwnedWeightPolicy.IsEligiblePointwise(consumer, n, k)) return false;
                }
                else
                {
                    if (source.Rank != 2) return false;
                    n = source.Dimensions[0]; k = source.Dimensions[1];
                    if (!ParakeetOwnedWeightPolicy.IsEligible(consumer, n, k)) return false;
                }
                if (!MemoryMarshal.TryGetArray<float>(source.Buffer, out var window) || window.Array is null
                    || window.Offset != 0 || window.Count != window.Array.Length
                    || protectedRoots.Contains(window.Array) || bindings[window.Array] != 1) return false;
                array = window.Array;
                return true;
            }
            // Only explicit owned preparation on the qualified host collapses
            // a region, and both pointwise payloads must pass the same ownership
            // admission as ordinary packed matrices before any node is removed.
            if (Avx512F.IsSupported && GraphFusion.FuseConformerConv(this, Optimization.GraphFacts.Build(this),
                (name, consumer) => Candidate(name, consumer, true, out _, out _, out _, out _)) != 0)
            {
                _prepared = false;
                EnsurePreparedLocked();
                uses = OwnedWeightUses(Nodes);
            }
            int converted = 0;
            // Names only: a values snapshot would unnecessarily retain every original array.
            foreach (string name in Initializers.Keys.ToArray())
            {
                if (!uses.TryGetValue(name, out var use) || use.ConsumerName is null
                    || !Candidate(name, use.ConsumerName, use.Pointwise, out var source, out var array, out int n, out int k)) continue;
                var replacement = use.Pointwise ? OwnedPackedTensor.FromPointwise(array, n, k)
                    : new OwnedPackedTensor(array, n, k);
                replacement.Name = source.Name;
                Initializers[name] = replacement;
                bindings.Remove(array);
                converted++;
                // Keep an executable graph even if a later allocation throws.
                _prepared = false;
            }
            if (converted != 0) ReleasedBuffers?.Clear();
            return converted;
        }
    }
}
