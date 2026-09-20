using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;
using Entry = Fingerprints.Entry;

Require(args.Length==4,"model output visit proof|timing");
string model=Path.GetFullPath(args[0]),output=Path.GetFullPath(args[1]);
int visit=int.Parse(args[2]);bool timing=args[3]=="timing";
Require(args[3] is "proof" or "timing" && visit>=0 && visit<4,"Arguments");
Require(!Directory.Exists(output),"Output exists");Directory.CreateDirectory(output);
long affinity=Affinity();
Require(Environment.ProcessorCount==1 && affinity==4,"Inherit CPU2 before CLR startup");
var settings=Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k=>k.StartsWith("LOKAD_",StringComparison.OrdinalIgnoreCase)||k.StartsWith("DOTNET_",StringComparison.OrdinalIgnoreCase)||k.StartsWith("COMPlus_",StringComparison.OrdinalIgnoreCase)).ToArray();
Require(settings.Length==0,"Runtime overrides");
const string core="187de61ad8f034b9b7ad2fb3490358443fa84334204720e81bc3546a31f3c8d4";
Require(Hash(typeof(ComputationalGraph).Assembly.Location)==core,"Qualified core");
var method=typeof(ComputationalGraph).GetMethod("ComputeStructureFingerprint",BindingFlags.NonPublic|BindingFlags.Instance,null,Type.EmptyTypes,null)!;
var actual=method.CreateDelegate<Func<ComputationalGraph,long>>();
int checks=0,cycles=0;var fixtures=new List<object>();
void Check(ComputationalGraph graph,Dictionary<ComputationalGraph,Entry[]> cache)
{
    long expected=actual(graph);
    Require(expected==Fingerprints.CopyA(graph)&&expected==Fingerprints.CopyB(graph)&&expected==Fingerprints.Cached(graph,cache),"Fingerprint bits");checks++;
}
Dictionary<ComputationalGraph,Entry[]> Train(ComputationalGraph graph)
{
    var cache=new Dictionary<ComputationalGraph,Entry[]>(ReferenceEqualityComparer.Instance);
    Require(Fingerprints.Train(graph,cache)==actual(graph),"Training bits");Check(graph,cache);return cache;
}
void Cycle(ComputationalGraph graph,Dictionary<ComputationalGraph,Entry[]> cache)
{
    foreach(var call in new Func<long>[] {()=>actual(graph),()=>Fingerprints.CopyA(graph),()=>Fingerprints.CopyB(graph),()=>Fingerprints.Cached(graph,cache)})
    {
        try {call();throw new InvalidDataException("Cycle accepted");}
        catch(InvalidOperationException e) {Require(e.Message=="Cyclic graph attributes are not supported.","Cycle message");cycles++;}
    }
}
string[] strings={"", "x", "same-name", "\0", "a\0b", "\u00e9", "\ud800", "\udfff", "\ud83d\ude80",new string('z',257)};
var random=new Random(20260920);
for(int test=0;test<40;test++)
{
    var graph=new ComputationalGraph();
    for(int i=0;i<1+test%9;i++)graph.Nodes.Add(new Node {Name=strings[(test+i)%strings.Length],Op=OpType.Add,Domain="",OpTypeName="Add",
        Inputs=new[]{"x","weight"},Outputs=new[]{"t"+i}});
    graph.Inputs["x"]=null!;graph.Initializers["weight"]=new DenseTensor<float>(new[]{1f},new[]{1});
    graph.InputDescs.Add(new OnnxValueInfo {Name="x"});graph.OutputDescs.Add(new OnnxValueInfo {Name="result"});
    var cache=Train(graph);int entries=cache.Values.Sum(a=>a.Length);var original=graph.Nodes.ToArray();
    for(int edit=0;edit<160;edit++)
    {
        int at=edit%graph.Nodes.Count;var node=graph.Nodes[at];string label=strings[random.Next(strings.Length)]+edit;
        switch(edit%16)
        {
            case 0:node.Name=label;break;
            case 1:node.Domain=label;break;
            case 2:node.OpTypeName=label;break;
            case 3:node.Op=OpType.Mul;break;
            case 4:node.Inputs=new[]{label,"weight"};break;
            case 5:node.Outputs=new[]{label,"other"};break;
            case 6:node.Inputs=null!;break;
            case 7:node.Outputs=null!;break;
            case 8:node.Name=null!;break;
            case 9:node.Attributes=new(){{"number",edit}};break;
            case 10:graph.Inputs["extra"+edit]=null!;break;
            case 11:graph.Initializers["extra"+edit]=graph.Initializers["weight"];break;
            case 12:graph.InputDescs[0].Name=label;break;
            case 13:graph.OutputDescs[0].Name=label;break;
            case 14:node.Name=new string((node.Name??"").ToCharArray());break;
            case 15:node.Inputs=new[]{"weight","x",label};break;
        }
        graph.Nodes[at]=node;Check(graph,cache);Require(cache.Values.Sum(a=>a.Length)==entries,"Cache grew on mutation");
        graph.Nodes[at]=original[at];graph.Inputs.Remove("extra"+edit);graph.Initializers.Remove("extra"+edit);
        graph.InputDescs[0].Name="x";graph.OutputDescs[0].Name="result";Check(graph,cache);
    }
    graph.Nodes.Reverse();Check(graph,cache);graph.Nodes.Reverse();
    graph.Nodes.Add(new Node {Name="new",Inputs=Array.Empty<string>(),Outputs=new[]{"new-output"}});Check(graph,cache);
    graph.Nodes.RemoveAt(graph.Nodes.Count-1);Check(graph,cache);
    fixtures.Add(new {name="flat-"+test,nodes=graph.Nodes.Select(n=>new {Name=Units(n.Name),op=(int)n.Op,Domain=Units(n.Domain),OpTypeName=Units(n.OpTypeName),Inputs=n.Inputs?.Select(Units).ToArray(),Outputs=n.Outputs?.Select(Units).ToArray()}).ToArray(),
        inputs=graph.Inputs.Keys.Select(Units).ToArray(),initializers=graph.Initializers.Keys.Select(Units).ToArray(),input_descriptions=graph.InputDescs.Select(x=>Units(x.Name)).ToArray(),
        output_descriptions=graph.OutputDescs.Select(x=>Units(x.Name)).ToArray(),fingerprint=actual(graph).ToString(),
        transitions=cache[graph].Select(e=>new {before=e.Before.ToString(),value=Units(e.Value),after=e.After.ToString()}).ToArray()});
}
var child=new ComputationalGraph();child.Nodes.Add(new Node {Name="child",Inputs=new[]{"capture"},Outputs=new[]{"result"}});child.Outputs["result"]=null!;
var parent=new ComputationalGraph();parent.Nodes.Add(new Node {Name="if",Inputs=new[]{"condition"},Outputs=new[]{"out"},Attributes=new(){{"then",child},{"else",child}}});
var nested=Train(parent);child.Outputs["new-read"]=null!;Check(parent,nested);child.Outputs.Remove("new-read");
child.Nodes[0].Inputs[0]="changed-capture";Check(parent,nested);
var added=new ComputationalGraph();added.Nodes.Add(new Node {Name="added",Inputs=null!,Outputs=new[]{"new"}});
parent.Nodes[0].Attributes!["third"]=added;Check(parent,nested);
added.Nodes.Add(new Node {Attributes=new(){{"parent",parent}}});Cycle(parent,nested);added.Nodes.RemoveAt(added.Nodes.Count-1);Check(parent,nested);
child.Nodes[0].Inputs[0]="capture";parent.Nodes[0].Attributes!.Remove("third");Check(parent,nested);
Parallel.For(0,128,_=>Require(actual(parent)==Fingerprints.Cached(parent,nested),"Concurrent readonly fingerprint"));
var untouched=actual(parent);var graphCount=nested.Count;Check(parent,nested);Require(actual(parent)==untouched&&nested.Count==graphCount,"Readonly structure");
long loadStart=Stopwatch.GetTimestamp();var e5=OnnxImport.Load(model)??throw new InvalidDataException("Model load");long loadTicks=Stopwatch.GetTimestamp()-loadStart;
e5.Prepare();long expectedE5=actual(e5);long trainBytes=GC.GetAllocatedBytesForCurrentThread(),trainStart=Stopwatch.GetTimestamp();
var e5cache=new Dictionary<ComputationalGraph,Entry[]>(ReferenceEqualityComparer.Instance);long trained=Fingerprints.Train(e5,e5cache);
long trainTicks=Stopwatch.GetTimestamp()-trainStart;trainBytes=GC.GetAllocatedBytesForCurrentThread()-trainBytes;
Require(trained==expectedE5,"Model training fingerprint");Check(e5,e5cache);
for(int i=0;i<e5.Nodes.Count;i++)
{
    var saved=e5.Nodes[i];var node=saved;node.Name=(node.Name??"")+"-changed";e5.Nodes[i]=node;Check(e5,e5cache);
    e5.Nodes[i]=saved;Check(e5,e5cache);
}
Require(actual(e5)==expectedE5,"Original model fingerprint changed");
var calls=new Func<long>[] {()=>actual(e5),()=>Fingerprints.CopyA(e5),()=>Fingerprints.CopyB(e5),()=>Fingerprints.Cached(e5,e5cache)};
int warm=timing?32:2,measured=timing?64:4,repeats=timing?256:8;var samples=new Sample[4*measured];int index=0;
for(int cycle=0;cycle<warm+measured;cycle++)for(int position=0;position<4;position++)
{
    int variant=(visit+cycle+position)%4;long observed=0;
    int g0=GC.CollectionCount(0),g1=GC.CollectionCount(1),g2=GC.CollectionCount(2);
    long allocated=GC.GetAllocatedBytesForCurrentThread(),start=Stopwatch.GetTimestamp();
    for(int i=0;i<repeats;i++)observed=calls[variant]();
    long ticks=Stopwatch.GetTimestamp()-start;allocated=GC.GetAllocatedBytesForCurrentThread()-allocated;
    Require(observed==expectedE5,"Timed fingerprint differs");
    if(cycle>=warm)samples[index++]=new(variant,cycle-warm,position,ticks,allocated,new[]{GC.CollectionCount(0)-g0,GC.CollectionCount(1)-g1,GC.CollectionCount(2)-g2});
}
Check(e5,e5cache);Require(index==samples.Length,"Sample coverage");
Write(Path.Combine(output,"fixtures.json"),fixtures);
Write(Path.Combine(output,"result.json"),new {schema=1,passed=true,timing,visit,checks,cycle_refusals=cycles,concurrent_checks=128,
    model_sha256=Hash(model),core_sha256=core,probe_sha256=Hash(Assembly.GetExecutingAssembly().Location),frequency=Stopwatch.Frequency,
    runtime=RuntimeInformation.FrameworkDescription,affinity=4,settings,load_ticks=loadTicks,training_ticks=trainTicks,training_allocated_bytes=trainBytes,
    cache_graphs=e5cache.Count,cache_entries=e5cache.Values.Sum(a=>a.Length),cache_struct_bytes=e5cache.Values.Sum(a=>a.LongLength)*24,
    nodes=e5.Nodes.Count,initializers=e5.Initializers.Count,fingerprint=expectedE5.ToString(),warmup_cycles=warm,measured_cycles=measured,repeats,samples,
    fixtures_sha256=Hash(Path.Combine(output,"fixtures.json")),scope="Repeated complete structure validation only; no inference or model speed ratio"});
Console.WriteLine($"PASS {checks} fingerprint checks; {cycles} cycle refusals; {index} retained batches; cache {e5cache.Values.Sum(a=>a.Length)} entries.");
static void Require(bool ok,string message){if(!ok)throw new InvalidDataException(message);}
static int[]? Units(string? text)=>text?.Select(c=>(int)c).ToArray();
static long Affinity()
{
    if(OperatingSystem.IsWindows()||OperatingSystem.IsLinux())return Process.GetCurrentProcess().ProcessorAffinity.ToInt64();
    throw new PlatformNotSupportedException();
}
static string Hash(string file){using var s=File.OpenRead(file);return Convert.ToHexStringLower(SHA256.HashData(s));}
static void Write(string path,object value){using var s=new FileStream(path,FileMode.CreateNew);JsonSerializer.Serialize(s,value,new JsonSerializerOptions{WriteIndented=true});}
readonly record struct Sample(int variant,int cycle,int position,long ticks,long bytes,int[] gc);
