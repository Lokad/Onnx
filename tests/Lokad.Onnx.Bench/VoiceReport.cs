using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using Lokad.Onnx;

namespace Lokad.Onnx.Bench;

// Maintained per-model execution report (W1): kernel routes, op inventory, top
// allocations and packing, as versioned JSON. No timing inside by design; the
// twin-judged bench owns latency. Demand/peak figures are cumulative since pool
// creation on a freshly loaded graph, so preparation is included.
public static class VoiceReport
{
    public static string Build(ComputationalGraph graph, Dictionary<string, ITensor> inputs, ExecutionOptions opts)
    {
        using var scope = Profiler.BeginExecution(true);
        if (!graph.Execute(inputs, true, ExecutionProvider.CPU, opts))
            throw new InvalidOperationException("report execution failed: " + graph.LastErrorMessage);
        var byId = new Dictionary<long, Node>();
        foreach (var n in graph.Nodes) byId[n.ID] = n;
        var nodes = new List<Dictionary<string, object?>>();
        var routes = new Dictionary<string, long>(StringComparer.Ordinal);
        var ops = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var np in graph.LastProfile!)
        {
            string nodeName = byId.TryGetValue(np.NodeId, out var nd) ? nd.Name : ("id" + np.NodeId);
            string route = "none";
            string inDescs = np.Detail;
            int ri = np.Detail.IndexOf(" route=", StringComparison.Ordinal);
            if (ri >= 0)
            {
                inDescs = np.Detail.Substring(0, ri);
                route = np.Detail.Substring(ri + 7);
                int sp = route.IndexOf(" ", StringComparison.Ordinal);
                if (sp >= 0) route = route.Substring(0, sp);
            }
            string op = np.Op.ToString();
            nodes.Add(new Dictionary<string, object?> { ["id"] = np.NodeId, ["name"] = nodeName, ["op"] = op, ["route"] = route, ["inputs"] = inDescs });
            routes[route] = routes.TryGetValue(route, out long a) ? a + 1 : 1;
            ops[op] = ops.TryGetValue(op, out long b) ? b + 1 : 1;
        }
        var alloc = new List<Dictionary<string, object>>();
        foreach (var d in graph.PoolDemandSnapshot().OrderByDescending(d => d.Missed * (long)d.Length).Take(10))
            alloc.Add(new Dictionary<string, object> { ["type"] = d.Type, ["length"] = d.Length, ["missed"] = d.Missed, ["reused"] = d.Reused });
        var doc = new Dictionary<string, object?>
        {
            ["nodes"] = nodes,
            ["routes"] = routes,
            ["ops"] = ops,
            ["topAlloc"] = alloc,
            ["peakLiveBytes"] = graph.LastPeakLiveBytes,
            ["poolPeakBytes"] = graph.LastPoolPeakOutstandingBytes,
            ["packingLegacy"] = graph.PackingReport.Live,
            ["packingLegacyBytes"] = graph.PackingReport.RetainedBytes,
            ["packingBlocked"] = graph.PackingReport.KBlockedLive,
            ["packingBlockedBytes"] = graph.PackingReport.KBlockedRetainedBytes,
        };
        return JsonSerializer.Serialize(doc, new JsonSerializerOptions { WriteIndented = true });
    }
}
