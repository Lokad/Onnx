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

    /// <summary>Replaces eligible float matrix initializers with independently owned packed storage.</summary>
    /// <returns>The number of initializers replaced; zero when no eligible weight or hardware is available.</returns>
    /// <remarks>
    /// Explicit opt-in for the supported Parakeet feed-forward matrix shapes and node names.
    /// Call while holding exclusive access to a newly loaded graph, before creating execution contexts.
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
            var uses = new Dictionary<string, (int Count, bool MatMulB)>(StringComparer.Ordinal);
            foreach (var node in Nodes)
            {
                var inputs = GraphCaptures.NodeInputs(node);
                for (int i = 0; i < inputs.Length; i++)
                {
                    string name = inputs[i]; if (string.IsNullOrEmpty(name)) continue;
                    bool eligible = node.Op == OpType.MatMul && Node.IsStandardDomain(node.Domain) && i == 1 && inputs.Length == 2
                        && node.Name?.Contains("/feed_forward", StringComparison.Ordinal) == true;
                    uses.TryGetValue(name, out var prior);
                    uses[name] = (prior.Count + 1, eligible && prior.Count == 0);
                }
            }
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
            int converted = 0;
            // Names only: a values snapshot would unnecessarily retain every original array.
            foreach (string name in Initializers.Keys.ToArray())
            {
                if (!uses.TryGetValue(name, out var use) || use.Count != 1 || !use.MatMulB
                    || Inputs.ContainsKey(name) || Outputs.ContainsKey(name) || folded.Contains(name)
                    || Initializers[name] is not DenseTensor<float> source || source.Rank != 2
                    || !GraphConvPacking.Standard(source)) continue;
                int n = source.Dimensions[0], k = source.Dimensions[1];
                if (!((n == 1024 && k == 4096) || (n == 4096 && k == 1024))) continue;
                if (!MemoryMarshal.TryGetArray<float>(source.Buffer, out var window) || window.Array is null
                    || window.Offset != 0 || window.Count != window.Array.Length
                    || protectedRoots.Contains(window.Array) || bindings[window.Array] != 1) continue;
                var replacement = new OwnedPackedTensor(window.Array, n, k) { Name = source.Name };
                Initializers[name] = replacement;
                bindings.Remove(window.Array);
                converted++;
                // Keep an executable graph even if a later allocation throws.
                _prepared = false;
                FingerprintStrings = null;
            }
            if (converted != 0) ReleasedBuffers?.Clear();
            return converted;
        }
    }
}
