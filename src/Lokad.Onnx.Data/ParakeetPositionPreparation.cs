namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

// Eagerly commute the supported static position slice with each linear
// projection. Used only on the newly loaded, private ParakeetTranscriber graph;
// the public API enforces encoded time <= maximumFrames. No mutable-core cache.
internal static class ParakeetPositionPreparation
{
    internal static int TryPrepare(ComputationalGraph graph, int maximumFrames)
    {
        if (maximumFrames is < 1 or > 376) return 0;
        var nodes = graph.Nodes;
        if (nodes.Any(n => n.Attributes?.Values.Any(v => v is ComputationalGraph) == true)) return 0;
        var produced = new Dictionary<string, Node>(StringComparer.Ordinal);
        foreach (var node in nodes)
            foreach (string output in node.Outputs)
                if (!string.IsNullOrEmpty(output) && !produced.TryAdd(output, node)) return 0;
        ITensor? Constant(string name)
        {
            if (graph.Initializers.TryGetValue(name, out var value)) return value;
            if (!produced.TryGetValue(name, out var node) || node.Op != OpType.Constant || !Node.IsStandardDomain(node.Domain)) return null;
            var result = CPUExecutionProvider.Constant(node.OneOfAttr("value", "value_int", "value_ints"), ExecutionOptions.Memory);
            return result.Status == OpStatus.Success && result.Outputs?.Length == 1 ? result.Outputs[0] : null;
        }
        bool Integer(string name, long expected) => Constant(name) is Tensor<long> value
            && value.Length == 1 && value.GetValue(0) == expected;
        bool Defined(string name, OpType op, int count, out Node node) =>
            produced.TryGetValue(name, out node) && Node.IsStandardDomain(node.Domain) && node.Op == op && node.Inputs.Length == count
            && (node.Attributes is null || node.Attributes.Count == 0
                || op == OpType.Gather && node.Attributes.Count == 1 && node.Attributes.ContainsKey("axis") && node.GetInt("axis", 0) == 0);
        bool Visible(string name) => graph.Inputs.ContainsKey(name) || graph.Outputs.ContainsKey(name);
        var slices = nodes.Where(n => n.Name == "/pos_enc/Slice").ToArray();
        if (slices.Length != 1) return 0;
        var slice = slices[0];
        if (!Node.IsStandardDomain(slice.Domain) || slice.Op != OpType.Slice || slice.Inputs.Length != 5 || slice.Outputs.Length != 1
            || slice.Attributes?.Count > 0 || Visible(slice.Outputs[0])
            || !Integer(slice.Inputs[3], 1) || !Integer(slice.Inputs[4], 1)
            || Constant(slice.Inputs[0]) is not Tensor<float> table
            || !table.Dimensions.SequenceEqual(new[] { 1, 9999, 1024 })
            || !Defined(slice.Inputs[1], OpType.Unsqueeze, 2, out var start)
            || !Defined(slice.Inputs[2], OpType.Unsqueeze, 2, out var end)
            || !Integer(start.Inputs[1], 0) || !Integer(end.Inputs[1], 0)
            || !Defined(start.Inputs[0], OpType.Sub, 2, out var subtract)
            || !Integer(subtract.Inputs[0], 5000)
            || !Defined(end.Inputs[0], OpType.Sub, 2, out var endMinusOne)
            || !Integer(endMinusOne.Inputs[1], 1)
            || !Defined(endMinusOne.Inputs[0], OpType.Add, 2, out var add)
            || !Integer(add.Inputs[0], 5000) || add.Inputs[1] != subtract.Inputs[1]) return 0;
        string frameCount = subtract.Inputs[1];
        if (!Defined(frameCount, OpType.Gather, 2, out var gather)
            || !Integer(gather.Inputs[1], 1)
            || !Defined(gather.Inputs[0], OpType.Shape, 1, out _)) return 0;
        var consumers = nodes.Where(n => n.Inputs.Contains(slice.Outputs[0])).ToArray();
        var projections = consumers.Where(n => n.Op == OpType.MatMul).ToArray();
        if (consumers.Any(n => n.Op != OpType.MatMul && (n.Op != OpType.Shape || n.Inputs.Length != 1
                || !Node.IsStandardDomain(n.Domain) || n.Attributes?.Count > 0))
            || projections.Length != 24 || nodes.Count(n => n.Inputs.Contains(slice.Inputs[0])) != 1
            || Visible(slice.Inputs[0])) return 0;
        var weights = new HashSet<string>(StringComparer.Ordinal);
        for (int layer = 0; layer < 24; layer++)
        {
            var expected = projections.Where(n => n.Name == $"/layers.{layer}/self_attn/linear_pos/MatMul").ToArray();
            if (expected.Length != 1) return 0;
            var node = expected[0];
            if (!Node.IsStandardDomain(node.Domain) || node.Op != OpType.MatMul || node.Inputs.Length != 2 || node.Inputs[0] != slice.Outputs[0]
                || node.Outputs.Length != 1 || node.Attributes?.Count > 0
                || !weights.Add(node.Inputs[1]) || Visible(node.Inputs[1])
                || nodes.Sum(n => n.Inputs.Count(i => i == node.Inputs[1])) != 1
                || !graph.Initializers.TryGetValue(node.Inputs[1], out var raw)
                || raw is not Tensor<float> weight || !weight.Dimensions.SequenceEqual(new[] { 1024, 1024 })) return 0;
        }
        const string prefix = "__lokad.parakeet.position.";
        var names = nodes.SelectMany(n => n.Inputs.Concat(n.Outputs)).Concat(graph.Initializers.Keys);
        if (names.Any(n => n.StartsWith(prefix, StringComparison.Ordinal))) return 0;
        int first = 5000 - maximumFrames, rows = 2 * maximumFrames - 1;
        static DenseTensor<long> Index(long value) => new(new[] { value }, new[] { 1 });
        static ITensor Result(OpResult result) => result.Status == OpStatus.Success && result.Outputs?.Length == 1
            ? result.Outputs[0] : throw new InvalidDataException("Static Parakeet position preparation failed.");
        var window = Result(CPUExecutionProvider.Slice(table, Index(first), Index(first + rows), Index(1), Index(1), ExecutionOptions.Memory));
        var prepared = new Dictionary<string, ITensor>(StringComparer.Ordinal);
        foreach (var node in projections)
        {
            var result = Result(CPUExecutionProvider.MatMul(window, graph.Initializers[node.Inputs[1]], ExecutionOptions.Memory, null));
            if (result is not Tensor<float> values || !values.Dimensions.SequenceEqual(new[] { 1, rows, 1024 })
                || values.ToArray().Any(v => !float.IsFinite(v))) throw new InvalidDataException("Nonfinite static position projection.");
            prepared.Add(node.Name, result);
        }
        // Commit only after complete admission and every projection succeeds.
        graph.InvalidatePreparation();
        graph.Initializers.Add(prefix + "offset", Index(first));
        long nextId = nodes.Max(n => n.ID) + 1;
        var rewritten = new List<Node>(nodes.Count);
        foreach (var original in nodes)
        {
            if (original.Outputs.Contains(slice.Inputs[0])) continue;
            if (original.Name == slice.Name)
            {
                rewritten.Add(new Node { ID = nextId++, Name = prefix + "start", Op = OpType.Sub, OpsetVersion = 13,
                    Inputs = new[] { slice.Inputs[1], prefix + "offset" }, Outputs = new[] { prefix + "starts" } });
                rewritten.Add(new Node { ID = nextId++, Name = prefix + "end", Op = OpType.Sub, OpsetVersion = 13,
                    Inputs = new[] { slice.Inputs[2], prefix + "offset" }, Outputs = new[] { prefix + "ends" } });
                // One shape-only consumer still needs the same [1,2*T-1,1024]
                // geometry. Reuse an already retained projected table for it.
                if (consumers.Length != projections.Length)
                {
                    var shapeSlice = original;
                    shapeSlice.Inputs = new[] { prefix + projections[0].Name, prefix + "starts", prefix + "ends", slice.Inputs[3], slice.Inputs[4] };
                    rewritten.Add(shapeSlice);
                }
                continue;
            }
            var node = original;
            if (prepared.TryGetValue(node.Name, out var projected))
            {
                string tableName = prefix + node.Name;
                graph.Initializers.Add(tableName, projected);
                graph.Initializers.Remove(node.Inputs[1]);
                node.Op = OpType.Slice;
                node.OpTypeName = "Slice";
                node.Inputs = new[] { tableName, prefix + "starts", prefix + "ends", slice.Inputs[3], slice.Inputs[4] };
            }
            rewritten.Add(node);
        }
        graph.Initializers.Remove(slice.Inputs[0]);
        graph.Nodes = rewritten;
        graph.Prepare();
        return prepared.Count;
    }
}
