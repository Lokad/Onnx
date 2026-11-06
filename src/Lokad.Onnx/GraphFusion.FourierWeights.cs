namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Linq;

internal static partial class GraphFusion
{
    internal static int FuseFourierWeights(ComputationalGraph graph, Optimization.GraphFacts facts)
    {
        var written=new HashSet<string>(StringComparer.Ordinal);
        foreach(var node in graph.Nodes)
        foreach(string output in node.Outputs)
            if(!string.IsNullOrEmpty(output)&&(!written.Add(output)||graph.Inputs.ContainsKey(output)||graph.Initializers.ContainsKey(output)))return 0;
        var removed=new HashSet<int>();int count=0;
        for(int final=0;final<graph.Nodes.Count;final++)
        {
            var last=graph.Nodes[final];
            if(last.Op!=OpType.Unsqueeze||!IsFusableParticipant(last)||last.OpsetVersion<13
                ||last.Inputs.Length!=2||last.Inputs.Any(string.IsNullOrEmpty)||last.Outputs.Length!=1
                ||string.IsNullOrEmpty(last.Outputs[0])||!Attributes(last))continue;
            int row=Producer(last.Inputs[0],OpType.Mul,2);if(row<0)continue;
            int window=Producer(graph.Nodes[row].Inputs[0],OpType.Mul,2);if(window<0)continue;
            int concat=Producer(graph.Nodes[window].Inputs[0],OpType.Concat,2,"axis");
            if(concat<0||graph.Nodes[concat].GetInt("axis",null)!=0)continue;
            int cosine=Producer(graph.Nodes[concat].Inputs[0],OpType.Cos,1);
            int negative=Producer(graph.Nodes[concat].Inputs[1],OpType.Neg,1);if(cosine<0||negative<0)continue;
            int sine=Producer(graph.Nodes[negative].Inputs[0],OpType.Sin,1);if(sine<0)continue;
            if(graph.Nodes[cosine].Inputs[0]!=graph.Nodes[sine].Inputs[0])continue;
            int phase=Producer(graph.Nodes[sine].Inputs[0],OpType.Mul,2);if(phase<0)continue;
            int cast=Producer(graph.Nodes[phase].Inputs[0],OpType.Cast,1,"to");
            if(cast<0||graph.Nodes[cast].GetInt("to",null)!=(int)TensorElementType.Float)continue;
            int mod=Producer(graph.Nodes[cast].Inputs[0],OpType.Mod,2,"fmod");
            if(mod<0||graph.Nodes[mod].GetInt("fmod",0)!=0)continue;
            int product=Producer(graph.Nodes[mod].Inputs[0],OpType.Mul,2);if(product<0)continue;
            int[] members=[row,window,concat,cosine,negative,sine,phase,cast,mod,product];
            if(members.Distinct().Count()!=10||members.Any(i=>i>=final||removed.Contains(i)))continue;
            var allowed=new HashSet<int>(members){final};
            if(allowed.Any(i=>graph.Nodes[i].Inputs.Any(name=>facts.Producer.TryGetValue(name,out int producer)&&producer>=i)))continue;
            if(members.Any(i=>graph.Nodes[i].Outputs.Any(name=>facts.GraphOutputs.Contains(name)||facts.CapturedValues.Contains(name)
                ||!facts.Consumers.TryGetValue(name,out var users)||users.Any(user=>!allowed.Contains(user)))))continue;
            string[] inputs=[graph.Nodes[product].Inputs[0],graph.Nodes[product].Inputs[1],graph.Nodes[mod].Inputs[1],
                graph.Nodes[phase].Inputs[1],graph.Nodes[window].Inputs[1],graph.Nodes[row].Inputs[1],last.Inputs[1]];
            TensorElementType[] types=[TensorElementType.Int64,TensorElementType.Int64,TensorElementType.Int64,
                TensorElementType.Float,TensorElementType.Float,TensorElementType.Float,TensorElementType.Int64];
            if(inputs.Where((name,i)=>!facts.Dtypes.TryGetValue(name,out var type)||type!=types[i]).Any())continue;
            // Every moved read must precede this recipe's original first use.
            // Global single assignment above prevents intervening name rebindings.
            if(inputs.Any(name=>facts.Producer.TryGetValue(name,out int index)&&index>=members.Min()))continue;
            last.Op=OpType.FusedFourierWeights;last.OpTypeName=nameof(OpType.FusedFourierWeights);
            last.Domain="";last.IsFused=true;last.Inputs=inputs;last.Attributes=null;
            graph.Nodes[final]=last;foreach(int i in members)removed.Add(i);count++;
        }
        if(removed.Count>0)
        {
            var kept=new List<Node>(graph.Nodes.Count-removed.Count);
            for(int i=0;i<graph.Nodes.Count;i++)if(!removed.Contains(i))kept.Add(graph.Nodes[i]);
            graph.Nodes.Clear();graph.Nodes.AddRange(kept);
        }
        return count;
        int Producer(string name,OpType op,int arity,params string[] attributes)
        {
            if(!facts.Producer.TryGetValue(name,out int index)||removed.Contains(index))return -1;
            var node=graph.Nodes[index];
            return node.Op==op&&IsFusableParticipant(node)&&node.Inputs.Length==arity&&node.Inputs.All(s=>!string.IsNullOrEmpty(s))
                &&node.Outputs.Length==1&&node.Outputs[0]==name&&Attributes(node,attributes)?index:-1;
        }
        static bool Attributes(Node node,params string[] allowed)=>node.Attributes is null||node.Attributes.Keys.All(allowed.Contains);
    }
}
