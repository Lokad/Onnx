using System.Runtime.InteropServices;
using System.Text.Json;
using Lokad.Onnx;
using Lokad.Onnx.Optimization;

namespace Lokad.Onnx.Backend.Tests;

public sealed class FourierWeightRecipeTests
{
    static bool Accelerated => System.Numerics.Vector.IsHardwareAccelerated && ExecutionOptions.Default.Tensor.UseSimd
        && ExecutionOptions.Default.Tensor.UseIntrinsics && ExecutionOptions.Default.Tensor.MaxDegreeOfParallelism == 1;
    static ComputationalGraph Graph(int length=16)
    {
        int bins=length/2+1;var graph=new ComputationalGraph();graph.Metadata["Name"]="fourier-generator";graph.Opset[""]=17;
        graph.Initializers["k"]=new DenseTensor<long>(Enumerable.Range(0,bins).Select(i=>(long)i).ToArray(),new[]{bins,1});
        graph.Initializers["n"]=new DenseTensor<long>(Enumerable.Range(0,length).Select(i=>(long)i).ToArray(),new[]{1,length});
        graph.Initializers["N"]=new DenseTensor<long>(new[]{(long)length},Array.Empty<int>());
        graph.Initializers["c"]=new DenseTensor<float>(new[]{(float)(2*Math.PI/length)},Array.Empty<int>());
        graph.Initializers["window"]=new DenseTensor<float>(Enumerable.Range(0,length).Select(i=>(float)(.5-.5*Math.Cos(2*Math.PI*i/length))).ToArray(),new[]{1,length});
        graph.Initializers["row"]=new DenseTensor<float>(Enumerable.Range(0,2*bins).Select(i=>i<bins?1f:.5f).ToArray(),new[]{2*bins,1});
        graph.Initializers["axes"]=new DenseTensor<long>(new[]{1L},new[]{1});
        static Node Node(OpType op,string[] inputs,string output,Dictionary<string,object>? attributes=null)=>new(){Op=op,OpTypeName=op.ToString(),Domain="",OpsetVersion=17,Inputs=inputs,Outputs=new[]{output},Attributes=attributes,Name=output};
        graph.Nodes.Add(Node(OpType.Mul,["k","n"],"kn"));graph.Nodes.Add(Node(OpType.Mod,["kn","N"],"mod"));
        graph.Nodes.Add(Node(OpType.Cast,["mod"],"cast",new(){["to"]=(int)TensorElementType.Float}));
        graph.Nodes.Add(Node(OpType.Mul,["cast","c"],"phase"));graph.Nodes.Add(Node(OpType.Cos,["phase"],"cos"));
        graph.Nodes.Add(Node(OpType.Sin,["phase"],"sin"));graph.Nodes.Add(Node(OpType.Neg,["sin"],"negative"));
        graph.Nodes.Add(Node(OpType.Concat,["cos","negative"],"concat",new(){["axis"]=0}));
        graph.Nodes.Add(Node(OpType.Mul,["concat","window"],"windowed"));graph.Nodes.Add(Node(OpType.Mul,["windowed","row"],"scaled"));
        graph.Nodes.Add(Node(OpType.Unsqueeze,["scaled","axes"],"weights"));graph.Outputs["weights"]=null!;
        return graph;
    }
    static ITensor Run(ComputationalGraph graph,ExecutionOptions? options=null,Dictionary<string,ITensor>? feed=null)
    {
        graph.Reset();Assert.True(graph.Execute(feed??new Dictionary<string,ITensor>(),feed is null,ExecutionProvider.CPU,options??ExecutionOptions.Default),graph.LastErrorMessage);
        return graph.Outputs["weights"]!;
    }
    static void Same(ITensor a,ITensor b)
    {
        Assert.Equal(a.Dims,b.Dims);Assert.True(MemoryMarshal.AsBytes(((Tensor<float>)a).ToArray().AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(((Tensor<float>)b).ToArray().AsSpan())));
    }
    [Fact]
    public void PreparedRecipeUsesLiveValuesAndPrivateOutputs()
    {
        var plain=Graph();var fused=Graph();Assert.Equal(1,GraphFusion.FuseFourierWeights(fused,GraphFacts.Build(fused)));fused.Prepare();
        Assert.Equal(Accelerated,fused.FourierWeights is not null);
        if(Accelerated)Assert.Single(fused.FourierWeights!);
        var held=Run(fused);var before=held.Clone();Same(Run(plain),held);
        string[] names=["k","n","N","c","window","row","axes"];
        foreach(string name in names)
        {
            ITensor a=plain.Initializers[name],b=fused.Initializers[name];var saved=a.Clone();
            if(a is Tensor<long> x&&b is Tensor<long> y){long value=name=="axes"?0:x.GetValue(0)+1;x.SetValue(0,value);y.SetValue(0,value);}
            else{var fx=(Tensor<float>)a;var fy=(Tensor<float>)b;float value=BitConverter.Int32BitsToSingle(BitConverter.SingleToInt32Bits(fx.GetValue(0))^1);fx.SetValue(0,value);fy.SetValue(0,value);}
            Same(Run(plain),Run(fused));Same(before,held);plain.Initializers[name]=saved.Clone();fused.Initializers[name]=saved.Clone();
        }
        foreach(var options in new[]{ExecutionOptions.Scalar,ExecutionOptions.Simd,ExecutionOptions.Default with{Tensor=ExecutionOptions.Default.Tensor with{MaxDegreeOfParallelism=2}}})Same(Run(plain,options),Run(fused,options));
        Same(Run(plain),Run(fused));Same(before,held);
    }
    [Theory]
    [InlineData("output")][InlineData("consumer")][InlineData("capture")][InlineData("domain")]
    [InlineData("attribute")][InlineData("cast")][InlineData("axis")][InlineData("fmod")]
    [InlineData("duplicate")][InlineData("order")][InlineData("opset")]
    public void MatcherPreservesObservableAndMalformedRecipes(string condition)
    {
        var graph=Graph();
        switch(condition)
        {
            case "output":graph.Outputs["phase"]=null!;break;
            case "consumer":graph.Nodes.Add(new(){Op=OpType.Identity,OpTypeName="Identity",OpsetVersion=17,Inputs=["phase"],Outputs=["extra"],Name="extra"});break;
            case "capture":
                var branch=new ComputationalGraph();branch.Outputs["phase"]=null!;
                graph.Nodes.Add(new(){Op=OpType.If,OpTypeName="If",OpsetVersion=17,Inputs=["condition"],Outputs=["extra"],Name="extra",Attributes=new(){["then_branch"]=branch,["else_branch"]=branch}});break;
            case "domain":var node=graph.Nodes[4];node.Domain="custom";graph.Nodes[4]=node;break;
            case "attribute":node=graph.Nodes[4];node.Attributes=new(){["extra"]=1};graph.Nodes[4]=node;break;
            case "cast":graph.Nodes[2].Attributes!["to"]=(int)TensorElementType.Double;break;
            case "axis":graph.Nodes[7].Attributes!["axis"]=1;break;
            case "fmod":node=graph.Nodes[1];node.Attributes=new(){["fmod"]=1};graph.Nodes[1]=node;break;
            case "duplicate":graph.Nodes.Insert(0,new(){Op=OpType.Identity,OpTypeName="Identity",OpsetVersion=17,Inputs=["n"],Outputs=["phase"],Name="duplicate"});break;
            case "order":(graph.Nodes[3],graph.Nodes[4])=(graph.Nodes[4],graph.Nodes[3]);break;
            case "opset":node=graph.Nodes[^1];node.OpsetVersion=11;graph.Nodes[^1]=node;break;
        }
        Assert.Equal(0,GraphFusion.FuseFourierWeights(graph,GraphFacts.Build(graph)));
    }
    [Fact]
    public void BudgetAndOverrideKeepPrimitiveSemantics()
    {
        var plain=Graph();var fused=Graph();Assert.Equal(1,GraphFusion.FuseFourierWeights(fused,GraphFacts.Build(fused)));
        foreach(var graph in new[]{plain,fused})graph.Inputs["row"]=graph.Initializers["row"];
        var replacement=((Tensor<float>)plain.Initializers["row"]).ToArray();replacement[0]=.25f;
        var feed=new Dictionary<string,ITensor>{{"row",new DenseTensor<float>(replacement,plain.Initializers["row"].Dims)}};
        Same(Run(plain,feed:feed),Run(fused,feed:feed));
        var inputs=new[]{"k","n","N","c","window","row","axes"}.Select(name=>fused.Initializers[name]).ToArray();
        var plan=PreparedFourierWeights.Create(inputs,long.MaxValue)!;Assert.NotNull(plan);
        Assert.Null(PreparedFourierWeights.Create(inputs,plan.RetainedBytes-1));
        var graphZero=new ComputationalGraph(0);graphZero.Metadata["Name"]="zero-budget";graphZero.Opset[""]=17;
        graphZero.Initializers=fused.Initializers;graphZero.Nodes=fused.Nodes;graphZero.Outputs["weights"]=null!;graphZero.Prepare();
        Assert.Null(graphZero.FourierWeights);Same(Run(plain),Run(graphZero));
    }
    [SkippableFact]
    public void OriginalInt8ModelUsesAllFourBoundedRecipes()
    {
        string? root=Environment.GetEnvironmentVariable("LONNX_PARADEE_FIXTURES");Skip.If(string.IsNullOrEmpty(root),"Set LONNX_PARADEE_FIXTURES for the original INT8 model admission check.");
        using var manifest=JsonDocument.Parse(File.ReadAllText(Path.Combine(root!,"manifest.json")));
        var spec=manifest.RootElement.GetProperty("models").EnumerateArray().Single(v=>v.GetProperty("variant").GetString()=="int8");
        var graph=OnnxImport.LoadOrThrow(Path.Combine(root!,spec.GetProperty("original").GetString()!));
        Assert.Equal(4,graph.Nodes.Count(n=>n.Op==OpType.FusedFourierWeights));graph.Prepare();
        Assert.Equal(Accelerated,graph.FourierWeights is not null);
        if(Accelerated)Assert.Equal(4,graph.FourierWeights!.Count);
        foreach(var node in graph.Nodes.Where(n=>n.Op==OpType.FusedFourierWeights))
        {
            var inputs=graph.GetInputTensors(node.Inputs);
            if(Accelerated){var plan=graph.FourierWeights![node.Outputs[0]];Assert.True(plan.Matches(inputs,ExecutionOptions.Default));Assert.True(plan.RetainedBytes>0);}
        }
        if(Accelerated)Assert.Equal(new[]{150,600},graph.FourierProjections!.Keys.Order());
        else Assert.Null(graph.FourierProjections);
    }
    [Fact]
    public void SharedContextsOwnOutputsAndInvalidationRebuildsCurrentValues()
    {
        var plain=Graph();var graph=Graph();Assert.Equal(1,GraphFusion.FuseFourierWeights(graph,GraphFacts.Build(graph)));graph.Prepare();
        var expected=Run(plain);var contexts=Enumerable.Range(0,4).Select(_=>graph.CreateExecution(ExecutionOptions.Default)).ToArray();
        var results=new ITensor[4];Parallel.For(0,4,i=>{Assert.True(contexts[i].Execute(new Dictionary<string,ITensor>(),true));results[i]=contexts[i].Outputs["weights"]!.Clone();Same(expected,results[i]);});
        ((Tensor<float>)graph.Initializers["window"]).SetValue(3,.123f);((Tensor<float>)plain.Initializers["window"]).SetValue(3,.123f);
        graph.InvalidatePreparation();graph.Prepare();Same(Run(plain),Run(graph));
        foreach(var value in results)Same(expected,value);
        var inputs=new[]{"k","n","N","c","window","row","axes"}.Select(name=>graph.Initializers[name]).ToArray();
        if(Accelerated)Assert.True(graph.FourierWeights!["weights"].Matches(inputs,ExecutionOptions.Default));
        else Assert.Null(graph.FourierWeights);
    }

    [Theory]
    [InlineData("window")][InlineData("row")][InlineData("c")]
    public void ChangedExceptionalValuesAndSignedZeroUseLivePrimitiveResults(string name)
    {
        var plain=Graph();var fused=Graph();Assert.Equal(1,GraphFusion.FuseFourierWeights(fused,GraphFacts.Build(fused)));fused.Prepare();
        foreach(float value in new[]{-0f,float.NaN,float.PositiveInfinity})
        {
            ((Tensor<float>)plain.Initializers[name]).SetValue(0,value);((Tensor<float>)fused.Initializers[name]).SetValue(0,value);
            Same(Run(plain),Run(fused));
        }
    }

    [Fact]
    public void NonDenseLiveWindowRetainsPrimitiveSemantics()
    {
        var plain=Graph();var fused=Graph();Assert.Equal(1,GraphFusion.FuseFourierWeights(fused,GraphFacts.Build(fused)));fused.Prepare();
        var values=((Tensor<float>)plain.Initializers["window"]).ToArray();var backing=new float[2*values.Length];
        for(int i=0;i<values.Length;i++)backing[2*i]=values[i];
        foreach(var graph in new[]{plain,fused})graph.Initializers["window"]=new TensorSlice<float>(new DenseTensor<float>(backing.ToArray(),new[]{1,backing.Length}),[SliceIndex.All,new SliceIndex(0,backing.Length,2)]);
        Same(Run(plain),Run(fused));
        var inputs=new[]{"k","n","N","c","window","row","axes"}.Select(name=>fused.Initializers[name]).ToArray();
        Assert.Null(PreparedFourierWeights.Create(inputs,long.MaxValue));
        if(fused.FourierWeights is { } held)Assert.False(held["weights"].Matches(inputs,ExecutionOptions.Default));
    }

    [Fact]
    public void CombinedPreparationBudgetAdmitsOnlyTheFirstCompleteRecipe()
    {
        var template=Graph();var inputs=new[]{"k","n","N","c","window","row","axes"}.Select(name=>template.Initializers[name]).ToArray();
        var bytes=PreparedFourierWeights.Create(inputs,long.MaxValue)!.RetainedBytes;
        var graph=new ComputationalGraph(2*bytes-1);graph.Metadata["Name"]="combined-budget";graph.Opset[""]=17;
        foreach(string prefix in new[]{"first/","second/"})
        {
            foreach(var pair in template.Initializers)graph.Initializers[prefix+pair.Key]=pair.Value.Clone();
            foreach(var node in template.Nodes)
            {
                var copy=node;copy.Name=prefix+node.Name;copy.Inputs=node.Inputs.Select(n=>prefix+n).ToArray();copy.Outputs=node.Outputs.Select(n=>prefix+n).ToArray();graph.Nodes.Add(copy);
            }
            graph.Outputs[prefix+"weights"]=null!;
        }
        Assert.Equal(2,GraphFusion.FuseFourierWeights(graph,GraphFacts.Build(graph)));graph.Prepare();
        Assert.Equal(Accelerated?bytes:0,graph.RetainedPackedWeightBytes);
        if(Accelerated){Assert.Single(graph.FourierWeights!);Assert.Contains("first/weights",graph.FourierWeights!.Keys);}
        else Assert.Null(graph.FourierWeights);
        Assert.True(graph.Execute(new Dictionary<string,ITensor>(),true),graph.LastErrorMessage);
        Same(graph.Outputs["first/weights"]!,graph.Outputs["second/weights"]!);
    }

    [Fact]
    public void PreparedOutputCopyIsReportedOnlyWhenAdmitted()
    {
        var graph=Graph();var inputs=new[]{"k","n","N","c","window","row","axes"}.Select(name=>graph.Initializers[name]).ToArray();
        var plan=PreparedFourierWeights.Create(inputs,long.MaxValue)!;Assert.NotNull(plan);
        var report=new CopyAccountant();var options=ExecutionOptions.Default with{Tensor=ExecutionOptions.Default.Tensor with{CopyReporter=report}};
        var output=plan.Execute(inputs,options,out bool admitted,new TensorBufferPool());
        Assert.Equal(Accelerated,admitted);Assert.Equal(admitted?output.Length*4:0,report.TotalCopyBytes);
        Same(PreparedFourierWeights.Primitives(inputs,ExecutionOptions.Default),output);
        var changed=inputs.Select(t=>t.Clone()).ToArray();((Tensor<float>)changed[5]).SetValue(0,.25f);
        var refusedReport=new CopyAccountant();options=options with{Tensor=options.Tensor with{CopyReporter=refusedReport}};
        var refused=plan.Execute(changed,options,out admitted,new TensorBufferPool());
        Assert.False(admitted);Assert.Equal(0,refusedReport.TotalCopyBytes);
        Same(PreparedFourierWeights.Primitives(changed,ExecutionOptions.Default),refused);
    }
}
