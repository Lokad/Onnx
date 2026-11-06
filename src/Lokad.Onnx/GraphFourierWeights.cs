namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Lokad.Onnx.Optimization;

internal static class GraphFourierWeights
{
    internal static IReadOnlyDictionary<string,PreparedFourierWeights>? Build(ComputationalGraph graph)
    {
        if(!graph.Options.Tensor.UseSimd||!graph.Options.Tensor.UseIntrinsics||!Vector.IsHardwareAccelerated
            ||graph.Options.Tensor.MaxDegreeOfParallelism!=1)return null;
        long remaining=Math.Min(graph.MaximumPackedWeightBytes,GraphPacking.MaxPackedBytes)-graph.RetainedPackedWeightBytes;
        if(remaining<=0)return null;
        var plans=new Dictionary<string,PreparedFourierWeights>(StringComparer.Ordinal);
        foreach(var node in graph.Nodes)
        {
            if(node.Op!=OpType.FusedFourierWeights||!node.IsFused||!Node.IsStandardDomain(node.Domain)
                ||node.Inputs.Length!=7||node.Outputs.Length!=1||string.IsNullOrEmpty(node.Outputs[0])
                ||plans.ContainsKey(node.Outputs[0])||node.Inputs.Any(name=>!graph.Initializers.ContainsKey(name)))continue;
            try
            {
                var plan=PreparedFourierWeights.Create(node.Inputs.Select(name=>graph.Initializers[name]).ToArray(),remaining);
                if(plan is null)continue;
                plans.Add(node.Outputs[0],plan);remaining-=plan.RetainedBytes;graph.RetainedPackedWeightBytes+=plan.RetainedBytes;
            }
            catch(Exception e) when(!Runtime.IsFatal(e)){ /* Unproved recipes retain primitive execution. */ }
        }
        return plans.Count==0?null:plans;
    }
}

public partial struct Node
{
    internal OpResult FourierWeightsPrepared(ComputationalGraph graph,ExecutionOptions options)
    {
        var inputs=graph.GetInputTensors(Inputs);
        if(graph.FourierWeights is { } plans&&plans.TryGetValue(Outputs[0],out var plan))
            return OpResult.Success(OpType.FusedFourierWeights,plan.Execute(inputs,options,out _,graph.ActivePool));
        return OpResult.Success(OpType.FusedFourierWeights,PreparedFourierWeights.Primitives(inputs,options));
    }
}
