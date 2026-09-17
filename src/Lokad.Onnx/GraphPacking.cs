namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// A MatMul weight initializer with a prepared panel-packed clone.
/// The packed payload is owned immutably by the prepared plan: replacing
/// the source initializer is detected per use and falls back to the
/// unpacked path, while structural edits require InvalidatePreparation,
/// which drops the clone like the folded transposes.
/// </summary>
/// <summary>
/// Prepared packing inventory for one graph: live panel-packed clone count,
/// retained clone bytes with correct element-size arithmetic, and live
/// counts by packed clone shape. Observability only; the admission policy
/// behind these numbers is unchanged.
/// </summary>
public sealed record PackedWeightsReport(
    int Live,
    long RetainedBytes,
    IReadOnlyList<PackedWeightShape> Shapes,
    int Eligible);

/// <summary>Live packed-clone count for one packed clone shape (rows by columns).</summary>
public sealed record PackedWeightShape(int Rows, int Cols, int Count);

/// <summary>
/// A row-major LSTM weight initializer with a prepared transposed clone.
/// The clone lays each [4H,K] direction slice out as [K,4H] so projections
/// read it directly; direction order never affects these bytes. Freshness
/// follows the packed-clone precedent: replacing the source initializer is
/// detected per use and falls back to the per-invocation build, while
/// structural edits require InvalidatePreparation.
/// </summary>
internal sealed record PreparedLstmTranspose(string SourceName, ITensor SourceRef, long SourceLength, DenseTensor<float> Transposed);

internal sealed record PackedMatMulWeight(
    string SourceName,
    ITensor SourceRef,
    long SourceLength,
    float[] SourceArray,
    string PackedName,
    DenseTensor<float> Packed);

/// <summary>
/// An LSTM input-weight (W) initializer with a prepared direction-major
/// panel-packed clone. Each direction slice holds PackPanelsB(inputSize, 4H)
/// of the transposed [inputSize,4H] slice, so the hoisted XW projection runs
/// the packed kernels with no per-run transpose or pack. Freshness follows
/// the packed-clone precedent: replacing the source initializer is detected
/// per use and falls back to the unpacked path, while structural edits
/// require InvalidatePreparation, which drops the clone like the others.
internal sealed record PreparedLstmPack(
    string SourceName,
    ITensor SourceRef,
    long SourceLength,
    float[] SourceArray,
    string PackedName,
    DenseTensor<float> Packed,
    int Directions,
    int InputSize,
    int FourH);

/// <summary>
/// Builds panel-packed clones of eligible MatMul weight initializers.
/// Dispatch resolves them through the per-graph mapping at call time, so
/// every fallback path keeps reading original row-major bytes by
/// construction. Follows the FoldedTransposes ownership precedent.
/// </summary>
internal static class GraphPacking
{
    internal const string PackedPrefix = "packed:";
    internal const string LstmTransposePrefix = "lstm-t:";
    internal const string LstmPackPrefix = "lstm-p:";

    /// <summary>Upper source-rows bound of measured packed-kernel territory: 4096 is admitted (encoder K=4096 projections at M=16), wider reductions are unmeasured and stay unpacked.</summary>
    internal const int MaxPackedAxis = 4096;

    /// <summary>Upper single-clone size in real bytes. Aggregate residency is intentionally not capped (see P2): capped subsets regressed the measured workload while full coverage wins, so the running total stays visible in PackingReport and bench headers.</summary>
    internal const long MaxPackedBytes = 512L * 1024 * 1024;

    /// <summary>Bytes of one n-by-k float32 packed clone with checked arithmetic.</summary>
    internal static long PackedCloneBytes(int n, int k) => checked((long)n * k * sizeof(float));

    /// <summary>
    /// Centralized per-matrix packing predicate: source rows within measured
    /// territory and a single clone inside the size bound, with exact byte math.
    /// </summary>
    internal static bool IsPackableShape(int n, int k)
    {
        if (n < 1 || k < 1 || n > MaxPackedAxis) return false;
        return PackedCloneBytes(n, k) <= MaxPackedBytes;
    }

    /// <summary>Resolves a MatMul edge name to a folded-transpose prepared tensor (P34).</summary>
    /// <remarks>Folded outputs materialize under a generated name, so the consuming edge name never hits Initializers directly; the prepared bytes are stable and guarded exactly like initializer sources downstream.</remarks>
    static bool TryResolveFoldedSource(ComputationalGraph graph, string edgeName, out ITensor? init)
    {
        init = null;
        foreach (var node in graph.Nodes)
        {
            if (node.Op != OpType.Transpose || node.Outputs is null) continue;
            bool produces = false;
            foreach (var o in node.Outputs) if (o == edgeName) { produces = true; break; }
            if (!produces) continue;
            if (graph.FoldedTransposes.TryGetValue(node.Name, out var fold)
                && graph.Initializers.TryGetValue(fold.PreparedName, out var prepared)
                && prepared is DenseTensor<float>)
            {
                init = prepared;
                return true;
            }
            return false;
        }
        return false;
    }
    /// <summary>
    /// Packs eligible MatMul B-side weight initializers of the graph.
    /// Eligible means float32, rank 2, source rows within measured territory with a single clone inside the byte budget, plain layout (Gemm transB must be 0),
    /// never a graph input or output, and consumed only as MatMul input 1.
    /// Fresh records reuse verified clones; stale records are dropped and
    /// rebuilt. Returns the live packed count.
    /// </summary>
    /// <summary>
    /// Prepares transposed clones of constant LSTM input-weight (W) and
    /// recurrent-weight (R) initializers. Each clone transposes every
    /// [4H,K] direction slice to [K,4H] once per preparation instead of
    /// once per invocation: the W clones serve the hoisted sequence-input
    /// projection, the R clones serve output-lane recurrent GEMV panels.
    /// Only direct float32 rank-three initializers with whole backing
    /// arrays qualify; anything else keeps the per-invocation build.
    /// Fresh records reuse verified clones; stale records are dropped and
    /// rebuilt. Returns the live prepared count.
    /// </summary>
    internal static int PrepareLstmWeights(ComputationalGraph graph)
    {
        var candidates = new HashSet<string>(StringComparer.Ordinal);
        foreach (var node in graph.Nodes)
        {
            if (node.Op != OpType.LSTM || node.Inputs is null || node.Inputs.Length < 3) continue;
            string input = node.Inputs[1];
            if (!string.IsNullOrEmpty(input)) candidates.Add(input);
            string recurrent = node.Inputs[2];
            if (!string.IsNullOrEmpty(recurrent)) candidates.Add(recurrent);
        }
        var current = new Dictionary<string, (ITensor tensor, float[] array)>(StringComparer.Ordinal);
        foreach (var name in candidates)
        {
            if (graph.Inputs.ContainsKey(name) || graph.Outputs.ContainsKey(name)) continue;
            if (!graph.Initializers.TryGetValue(name, out var init)) continue;
            if (init is not DenseTensor<float> dense || init.ElementType != TensorElementType.Float) continue;
            if (init.Rank != 3) continue;
            int[] dims = init.Dims;
            if (dims.Length != 3) continue;
            int d = dims[0], gh = dims[1], k = dims[2];
            if (d < 1 || gh < 4 || k < 1 || gh % 4 != 0) continue;
            if (!System.Runtime.InteropServices.MemoryMarshal.TryGetArray(dense.Buffer, out System.ArraySegment<float> window)
                || window.Array is null || window.Offset != 0 || window.Count != dense.Buffer.Length) continue;
            current[name] = (init, window.Array);
        }
        int live = 0;
        var stale = new List<float[]>();
        foreach (var kv in graph.LstmTransposes)
        {
            var rec = kv.Value;
            if (current.TryGetValue(rec.SourceName, out var cur)
                && ReferenceEquals(cur.tensor, rec.SourceRef) && cur.tensor.Length == rec.SourceLength)
            {
                live++;
                continue;
            }
            stale.Add(kv.Key);
        }
        foreach (float[] key in stale)
        {
            if (graph.LstmTransposes.TryGetValue(key, out var rec))
            {
                if (graph.Initializers.TryGetValue(rec.Transposed.Name, out var held) && ReferenceEquals(held, rec.Transposed))
                    graph.Initializers.Remove(rec.Transposed.Name);
                graph.LstmTransposes.Remove(key);
            }
        }
        foreach (var kv in current)
        {
            bool already = false;
            foreach (var rec in graph.LstmTransposes.Values)
            {
                if (ReferenceEquals(rec.SourceRef, kv.Value.tensor)) { already = true; break; }
            }
            if (already) continue;
            string preparedName = LstmTransposePrefix + kv.Key;
            if (graph.Initializers.ContainsKey(preparedName) || graph.Inputs.ContainsKey(preparedName)) continue;
            var dense = (DenseTensor<float>)kv.Value.tensor;
            int dd = dense.Dimensions[0], ggh = dense.Dimensions[1], kk = dense.Dimensions[2];
            var panel = new float[(long)dd * ggh * kk];
            unsafe
            {
                using var sh = dense.Buffer.Pin();
                using var ph = new Memory<float>(panel).Pin();
                float* sp = (float*)sh.Pointer;
                float* dp = (float*)ph.Pointer;
                for (int dir = 0; dir < dd; dir++)
                    for (int row = 0; row < ggh; row++)
                        for (int col = 0; col < kk; col++)
                            dp[(dir * kk + col) * ggh + row] = sp[(dir * ggh + row) * kk + col];
            }
            var transposed = new DenseTensor<float>(new Memory<float>(panel), new int[] { dd, kk, ggh });
            transposed.Name = preparedName;
            graph.Initializers[preparedName] = transposed;
            graph.LstmTransposes[kv.Value.array] = new PreparedLstmTranspose(kv.Key, kv.Value.tensor, kv.Value.tensor.Length, transposed);
            live++;
        }
        return live;
    }

    /// <summary>
    /// Resolves an LSTM weight operand to its prepared transposed clone when
    /// the unwrapped source is the unchanged initializer the clone was built
    /// from. Broadcast views resolve through their source; offset views and
    /// replaced initializers fall back to the per-invocation build. The clone
    /// holds every direction slice, so callers window per-direction views.
    /// </summary>
    internal static DenseTensor<float>? ResolveLstmTranspose(IReadOnlyDictionary<float[], PreparedLstmTranspose>? map, Tensor<float> source)
    {
        if (map is null || map.Count == 0) return null;
        Tensor<float> core = source;
        while (core is BroadcastedTensor<float> view) core = view.source;
        if (core is DenseTensor<float> dense
            && DenseArray(dense) is float[] backing
            && map.TryGetValue(backing, out var rec)
            && ReferenceEquals(rec.SourceRef, dense)
            && rec.SourceLength == dense.Length
            && rec.Transposed.Length == dense.Length)
        {
            return rec.Transposed;
        }
        return null;
    }

    /// <summary>
    /// Prepares direction-major panel-packed clones of constant LSTM
    /// input-weight (W) initializers. Admission mirrors PrepareLstmWeights
    /// (float32 rank-three [directions,4H,inputSize] whole-array
    /// initializers); the packed clone additionally requires packable panel
    /// dims. Each direction slice packs the transposed [inputSize,4H] slice
    /// once per preparation instead of paying Densify plus the unpacked lane
    /// on every invocation. Fresh records reuse verified clones; stale
    /// records are dropped and rebuilt. Returns the live prepared count.
    /// </summary>
    internal static int PrepareLstmPacks(ComputationalGraph graph)
    {
        var candidates = new HashSet<string>(StringComparer.Ordinal);
        foreach (var node in graph.Nodes)
        {
            if (node.Op != OpType.LSTM || node.Inputs is null || node.Inputs.Length < 3) continue;
            string input = node.Inputs[1];
            if (!string.IsNullOrEmpty(input)) candidates.Add(input);
        }
        var current = new Dictionary<string, (ITensor tensor, float[] array)>(StringComparer.Ordinal);
        foreach (var name in candidates)
        {
            if (graph.Inputs.ContainsKey(name) || graph.Outputs.ContainsKey(name)) continue;
            if (!graph.Initializers.TryGetValue(name, out var init)) continue;
            if (init is not DenseTensor<float> dense || init.ElementType != TensorElementType.Float) continue;
            if (init.Rank != 3) continue;
            int[] dims = init.Dims;
            if (dims.Length != 3) continue;
            int d = dims[0], gh = dims[1], k = dims[2];
            if (d < 1 || gh < 4 || k < 1 || gh % 4 != 0) continue;
            // Packed XW panels read [inputSize,4H] per direction; require the
            // same packable panel dims as the MatMul clones.
            if (!IsPackableShape(k, gh)) continue;
            if (!System.Runtime.InteropServices.MemoryMarshal.TryGetArray(dense.Buffer, out System.ArraySegment<float> window)
                || window.Array is null || window.Offset != 0 || window.Count != dense.Buffer.Length) continue;
            current[name] = (init, window.Array);
        }
        int live = 0;
        var stale = new List<float[]>();
        foreach (var kv in graph.LstmPacks)
        {
            var rec = kv.Value;
            if (current.TryGetValue(rec.SourceName, out var cur)
                && ReferenceEquals(cur.tensor, rec.SourceRef) && cur.tensor.Length == rec.SourceLength
                && ReferenceEquals(cur.array, rec.SourceArray))
            {
                live++;
                continue;
            }
            stale.Add(kv.Key);
        }
        foreach (float[] key in stale)
        {
            if (graph.LstmPacks.TryGetValue(key, out var rec))
            {
                if (graph.Initializers.TryGetValue(rec.PackedName, out var held) && ReferenceEquals(held, rec.Packed))
                    graph.Initializers.Remove(rec.PackedName);
                graph.LstmPacks.Remove(key);
            }
        }
        foreach (var kv in current)
        {
            bool already = false;
            foreach (var rec in graph.LstmPacks.Values)
            {
                if (ReferenceEquals(rec.SourceRef, kv.Value.tensor)) { already = true; break; }
            }
            if (already) continue;
            string packedName = LstmPackPrefix + kv.Key;
            if (graph.Initializers.ContainsKey(packedName) || graph.Inputs.ContainsKey(packedName)) continue;
            var dense = (DenseTensor<float>)kv.Value.tensor;
            int dd = dense.Dimensions[0], ggh = dense.Dimensions[1], kk = dense.Dimensions[2];
            var panel = new float[(long)dd * ggh * kk];
            unsafe
            {
                using var sh = dense.Buffer.Pin();
                using var ph = new Memory<float>(panel).Pin();
                float* sp = (float*)sh.Pointer;
                float* pp = (float*)ph.Pointer;
                // Transpose each direction slice to [inputSize,4H], then panel-pack it.
                var tmp = new float[(long)kk * ggh];
                fixed (float* tp = tmp)
                {
                    for (int dir = 0; dir < dd; dir++)
                    {
                        for (int row = 0; row < ggh; row++)
                            for (int col = 0; col < kk; col++)
                                tp[col * ggh + row] = sp[(dir * ggh + row) * kk + col];
                        MathOps.PackPanelsB(kk, ggh, tp, pp + dir * kk * ggh);
                    }
                }
            }
            var packed = new DenseTensor<float>(new Memory<float>(panel), new int[] { dd, kk, ggh });
            packed.Name = packedName;
            graph.Initializers[packedName] = packed;
            graph.LstmPacks[kv.Value.array] = new PreparedLstmPack(kv.Key, kv.Value.tensor, kv.Value.tensor.Length, kv.Value.array, packedName, packed, dd, kk, ggh);
            live++;
        }
        return live;
    }

    /// <summary>
    /// Resolves an LSTM input-weight operand to its prepared panel-packed
    /// clone when the unwrapped source is the unchanged initializer the clone
    /// was built from. Broadcast views resolve through their source; offset
    /// views and replaced initializers fall back to the unpacked path.
    /// Callers window per-direction packed slices from the whole clone.
    /// </summary>
    internal static DenseTensor<float>? ResolveLstmPack(IReadOnlyDictionary<float[], PreparedLstmPack>? map, Tensor<float> source)
    {
        if (map is null || map.Count == 0) return null;
        Tensor<float> core = source;
        while (core is BroadcastedTensor<float> view) core = view.source;
        if (core is DenseTensor<float> dense
            && DenseArray(dense) is float[] backing
            && map.TryGetValue(backing, out var rec)
            && ReferenceEquals(rec.SourceRef, dense)
            && rec.SourceLength == dense.Length
            && rec.Packed.Length == dense.Length)
        {
            return rec.Packed;
        }
        return null;
    }

    internal static int PackMatMulWeights(ComputationalGraph graph)
    {
        var consumers = new Dictionary<string, bool>(StringComparer.Ordinal);
        foreach (var node in graph.Nodes)
        {
            if (node.Inputs is null) continue;
            for (int i = 0; i < node.Inputs.Length; i++)
            {
                string input = node.Inputs[i];
                if (string.IsNullOrEmpty(input)) continue;
                bool eligible = (node.Op == OpType.MatMul && i == 1) || (node.Op == OpType.Gemm && i == 1 && (node.GetInt("transB", 0) ?? 0) == 0);
                if (consumers.TryGetValue(input, out bool prior)) consumers[input] = prior && eligible;
                else consumers[input] = eligible;
            }
        }
        var current = new Dictionary<string, (ITensor tensor, float[] array)>(StringComparer.Ordinal);
        foreach (var kv in consumers)
        {
            if (!kv.Value) continue;
            if (graph.Inputs.ContainsKey(kv.Key) || graph.Outputs.ContainsKey(kv.Key)) continue;
            if (!graph.Initializers.TryGetValue(kv.Key, out var init) && !TryResolveFoldedSource(graph, kv.Key, out init)) continue;
            if (init is not DenseTensor<float> dense || init.ElementType != TensorElementType.Float) continue;
            if (init.Rank != 2 || dense.IsReversedStride) continue;
            if (!dense.strides.SequenceEqual(ArrayUtilities.GetStrides(dense.dimensions))) continue;
            int[] dims = init.Dims;
            if (dims.Length != 2) continue;
            int n = dims[0], k = dims[1];
            if (!IsPackableShape(n, k)) continue;
            if (!System.Runtime.InteropServices.MemoryMarshal.TryGetArray(dense.Buffer, out System.ArraySegment<float> window)
                || window.Array is null || window.Offset != 0 || window.Count != dense.Buffer.Length) continue;
            current[kv.Key] = (init, window.Array);
        }
        int live = 0;
        var stale = new List<float[]>();
        foreach (var kv in graph.PackedWeights)
        {
            var rec = kv.Value;
            if (current.TryGetValue(rec.SourceName, out var cur)
                && ReferenceEquals(cur.tensor, rec.SourceRef) && cur.tensor.Length == rec.SourceLength
                && ReferenceEquals(cur.array, rec.SourceArray)
                && graph.Initializers.TryGetValue(rec.PackedName, out var held)
                && ReferenceEquals(held, rec.Packed))
            {
                live++;
                continue;
            }
            stale.Add(kv.Key);
        }
        foreach (float[] key in stale)
        {
            if (graph.PackedWeights.TryGetValue(key, out var rec))
            {
                if (graph.Initializers.TryGetValue(rec.PackedName, out var held) && ReferenceEquals(held, rec.Packed))
                    graph.Initializers.Remove(rec.PackedName);
                graph.PackedWeights.Remove(key);
            }
        }
        foreach (var kv in current)
        {
            bool already = false;
            foreach (var rec in graph.PackedWeights.Values)
            {
                if (rec.SourceName == kv.Key && ReferenceEquals(rec.SourceRef, kv.Value.tensor)) { already = true; break; }
            }
            if (already) continue;
            string packedName = PackedPrefix + kv.Key;
            if (graph.Initializers.ContainsKey(packedName) || graph.Inputs.ContainsKey(packedName)) continue;
            int n = kv.Value.tensor.Dims[0], k = kv.Value.tensor.Dims[1];
            var panel = new float[(long)n * k];
            var dense = (DenseTensor<float>)kv.Value.tensor;
            unsafe
            {
                using var sh = dense.Buffer.Pin();
                using var ph = new Memory<float>(panel).Pin();
                MathOps.PackPanelsB(n, k, (float*)sh.Pointer, (float*)ph.Pointer);
            }
            var packed = new DenseTensor<float>(new Memory<float>(panel), new int[] { n, k });
            packed.Name = packedName;
            graph.Initializers[packedName] = packed;
            graph.PackedWeights[kv.Value.array] = new PackedMatMulWeight(kv.Key, kv.Value.tensor, kv.Value.tensor.Length, kv.Value.array, packedName, packed);
            live++;
        }
        long retained = 0;
        var byShape = new Dictionary<(int Rows, int Cols), int>();
        foreach (var rec in graph.PackedWeights.Values)
        {
            int rn = rec.Packed.Dimensions[0];
            int rk = rec.Packed.Dimensions[1];
            checked { retained += PackedCloneBytes(rn, rk); }
            var key = (rn, rk);
            byShape.TryGetValue(key, out int shaped);
            byShape[key] = shaped + 1;
        }
        var shapes = byShape
            .OrderBy(kv => kv.Key.Rows)
            .ThenBy(kv => kv.Key.Cols)
            .Select(kv => new PackedWeightShape(kv.Key.Rows, kv.Key.Cols, kv.Value))
            .ToArray();
        graph.PackingReport = new PackedWeightsReport(graph.PackedWeights.Count, retained, shapes, current.Count);
        return live;
    }
    /// <summary>
    /// Resolves a B-side operand to its packed clone when the unwrapped
    /// source is a fresh mapped weight. Broadcast views resolve through
    /// their source; anything else falls back to the unpacked path.
    /// </summary>
    internal static DenseTensor<float>? ResolvePacked(IReadOnlyDictionary<float[], PackedMatMulWeight>? map, Tensor<float> y)
    {
        if (map is null || map.Count == 0) return null;
        Tensor<float> core = y;
        while (core is BroadcastedTensor<float> view) core = view.source;
        // Offset views (for example per-direction LSTM weight windows over one
        // shared transpose buffer) expose no whole backing array and must fall
        // back to the unpacked path instead of throwing on a null key.
        if (core is DenseTensor<float> dense
            && DenseArray(dense) is float[] backing
            && map.TryGetValue(backing, out var rec)
            && ReferenceEquals(rec.SourceArray, DenseArray(dense))
            && TrailingDimsMatch(y, rec.Packed))
        {
            return rec.Packed;
        }
        Tensor<float> direct = y;
        while (direct is BroadcastedTensor<float> vw) direct = vw.source;
        if (direct is DenseTensor<float> named && named.Name is not null && named.Name.StartsWith(PackedPrefix, StringComparison.Ordinal))
        {
            foreach (var cand in map.Values)
            {
                if (cand.PackedName == named.Name && ReferenceEquals(cand.Packed, named)) return cand.Packed;
            }
        }
        return null;
    }

    static float[]? DenseArray(DenseTensor<float> dense)
    {
        if (!System.Runtime.InteropServices.MemoryMarshal.TryGetArray(dense.Buffer, out System.ArraySegment<float> window)
            || window.Array is null || window.Offset != 0) return null;
        return window.Array;
    }
    /// <summary>
    /// The operand trailing two dims must equal the packed dims, so the
    /// packed kernel reads the panels it was built from.
    /// </summary>
    static bool TrailingDimsMatch(Tensor<float> y, DenseTensor<float> packed)
    {
        int r = y.Rank;
        if (r < 2) return false;
        int[] pd = packed.Dimensions.ToArray();
        if (pd.Length != 2) return false;
        return y.Dimensions[r - 2] == pd[0] && y.Dimensions[r - 1] == pd[1];
    }
}
