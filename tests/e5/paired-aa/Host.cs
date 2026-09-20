using System.Diagnostics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text.Json;
using Call = System.Func<(long Execute, long Request, long Bytes, int G0, int G1, int G2)>;

if(args.Length!=6) throw new ArgumentException("model case.json output-directory visit case-index smoke|timing");
string model=Path.GetFullPath(args[0]), caseFile=Path.GetFullPath(args[1]), output=Path.GetFullPath(args[2]);
int visit=int.Parse(args[3]), caseIndex=int.Parse(args[4]); bool smoke=args[5]=="smoke";
Require(args[5] is "smoke" or "timing", "Mode"); Require(!Directory.Exists(output), "Output exists");
Require(visit>=0 && visit<4 && caseIndex>=0 && caseIndex<5, "Schedule position");
var settings=Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k=>k.StartsWith("LOKAD_",StringComparison.OrdinalIgnoreCase)
    || k.StartsWith("DOTNET_",StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_",StringComparison.OrdinalIgnoreCase)).ToArray();
Require(settings.Length==0, "Runtime overrides present"); NoOrt();
Require(Environment.ProcessorCount==1 && Affinity()==4, "Inherited CPU2 required");
Directory.CreateDirectory(output);
int pairs=smoke?4:64, solo=smoke?4:32; double seconds=smoke?1:30;
int[] creation=visit%2==0 ? new[]{0,1}:new[]{1,0};
var contexts=new PrivateContext[2]; var cores=new Assembly[2]; var bridgeTypes=new Type[2];
var initialize=new Func<string,string,string,string>[2]; var calls=new Call[2]; var finish=new Func<string>[2];
foreach(int arm in creation)
{
    var context=new PrivateContext("arm"+arm,AppContext.BaseDirectory); contexts[arm]=context;
    cores[arm]=context.LoadFromAssemblyPath(Path.Combine(AppContext.BaseDirectory,"Lokad.Onnx.dll"));
    var bridge=context.LoadFromAssemblyPath(Path.Combine(AppContext.BaseDirectory,"PairedBridge.dll"));
    var type=bridge.GetType("PairedBridge",true)!;bridgeTypes[arm]=type;
    initialize[arm]=type.GetMethod("Initialize")!.CreateDelegate<Func<string,string,string,string>>();
    calls[arm]=type.GetMethod("Run")!.CreateDelegate<Call>(); finish[arm]=type.GetMethod("Finish")!.CreateDelegate<Func<string>>();
}
Require(!ReferenceEquals(cores[0],cores[1]) && bridgeTypes[0]!=bridgeTypes[1], "Assemblies shared");
var switchTypes=cores.Select(a=>a.GetType("Lokad.Onnx.AblationSwitches",true)!).ToArray();
foreach(var type in switchTypes)RuntimeHelpers.RunClassConstructor(type.TypeHandle);
var latches=switchTypes.Select(t=>t.GetField("LegacySoftmaxUsed",BindingFlags.Static|BindingFlags.NonPublic)!).ToArray();
Require(latches.All(f=>f.GetValue(null) is false), "Initial latch");
latches[0].SetValue(null,true);
try { Require(latches[0].GetValue(null) is true && latches[1].GetValue(null) is false, "Static state shared"); }
finally { latches[0].SetValue(null,false); }
var flags=switchTypes.Select(t=>t.GetFields(BindingFlags.Static|BindingFlags.NonPublic).Where(f=>f.FieldType==typeof(bool))
    .OrderBy(f=>f.Name,StringComparer.Ordinal).ToDictionary(f=>f.Name,f=>(bool)f.GetValue(null)!)).ToArray();
Require(JsonSerializer.Serialize(flags[0])==JsonSerializer.Serialize(flags[1]), "A/A flags differ");
Require(!flags[0]["EnableBiasGeluInterleaved"], "Candidate present in A/A");
int[] orders=Enumerable.Range(0,pairs).Select(i=>i%2).ToArray();
uint state=checked((uint)(20260920+100*visit+caseIndex));
for(int i=orders.Length-1;i>0;i--) { state=unchecked(state*1664525+1013904223);int j=(int)(state%(uint)(i+1));(orders[i],orders[j])=(orders[j],orders[i]); }
var schedule=new {schema=1, protocol="paired-managed-aa-v1", visit, case_index=caseIndex, smoke,
    creation, pair_first=orders, solo_order=creation, solo_before_pairs=visit%2!=0, conditioning_seconds=seconds,pairs,solo};
Write(Path.Combine(output,"schedule.json"),schedule);
var initial=new JsonElement[2];var final=new JsonElement[2];
foreach(int arm in creation)initial[arm]=JsonSerializer.Deserialize<JsonElement>(initialize[arm](model,caseFile,Path.Combine(output,"arm"+arm)));
Require(!AssemblyLoadContext.Default.Assemblies.Any(a=>a.GetName().Name=="Lokad.Onnx"), "Core leaked into default context");
Require(contexts.All(c=>c.Assemblies.Single(a=>a.GetName().Name=="Lokad.Onnx")!=null), "Missing private core");
var conditioning=new List<Row>(20000);var measured=new Row[2*pairs+2*solo];int cursor=0;
double[] conditioned={0,0};int round=0;
while(conditioned.Any(x=>x<seconds))
{
    foreach(int arm in creation)
    {
        if(conditioned[arm]>=seconds)continue;
        Require(conditioning.Count<20000,"Conditioning capacity");
        var row=Take(arm,"conditioning",round,Array.IndexOf(creation,arm));conditioning.Add(row);
        conditioned[arm]+=(double)row.execute/Stopwatch.Frequency;
    }
    round++;
}
Console.WriteLine("Conditioned both engines: "+Path.GetFileName(caseFile));
var before=Resources();
if(visit%2!=0)Solo();
for(int pair=0;pair<pairs;pair++)for(int position=0;position<2;position++)
    measured[cursor++]=Take(position==0?orders[pair]:1-orders[pair],"paired",pair,position);
if(visit%2==0)Solo();
var after=Resources();Require(cursor==measured.Length,"Coverage");
foreach(int arm in creation)final[arm]=JsonSerializer.Deserialize<JsonElement>(finish[arm]());
Require(final[0].GetProperty("output_sha256").GetString()==final[1].GetProperty("output_sha256").GetString(),"A/A output bits differ");
Require(latches.All(f=>f.GetValue(null) is false),"Latch changed");NoOrt();
Write(Path.Combine(output,"result.json"),new {schema=1,protocol="paired-managed-aa-v1", schedule,
    frequency=Stopwatch.Frequency,affinity=Affinity(),runtime=RuntimeInformation.FrameworkDescription,
    avx2=Avx2.IsSupported,avx512=Avx512F.IsSupported,settings,flags,static_isolation=true,private_cores=true,
    host_sha256=Hash(Assembly.GetExecutingAssembly().Location),case_sha256=Hash(caseFile),initial,final,
    conditioning,conditioned,measured,before,after,passed=true});
Console.WriteLine("Completed paired A/A "+Path.GetFileName(caseFile)+" visit "+visit);

Row Take(int arm,string phase,int group,int position)
{
    var t=calls[arm]();return new Row(arm,phase,group,position,t.Execute,t.Request,t.Bytes,t.G0,t.G1,t.G2);
}
void Solo() { foreach(int arm in creation)for(int i=0;i<solo;i++)measured[cursor++]=Take(arm,"solo",i,Array.IndexOf(creation,arm)); }
static void Require(bool ok,string message) { if(!ok)throw new InvalidDataException(message); }
static string Hash(string path) { using var f=File.OpenRead(path);return Convert.ToHexStringLower(SHA256.HashData(f)); }
static long Affinity()
{
    if(OperatingSystem.IsWindows() || OperatingSystem.IsLinux())return Process.GetCurrentProcess().ProcessorAffinity.ToInt64();
    throw new PlatformNotSupportedException();
}
static void NoOrt() => Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m=>Path.GetFileName(m.FileName).StartsWith("libonnxruntime",StringComparison.OrdinalIgnoreCase)
    ||Path.GetFileName(m.FileName).Equals("onnxruntime.dll",StringComparison.OrdinalIgnoreCase)),"Native ORT loaded");
static object Resources() { using var p=Process.GetCurrentProcess();var m=GC.GetGCMemoryInfo();return new {rss=p.WorkingSet64,peak=p.PeakWorkingSet64,heap=m.HeapSizeBytes,committed=m.TotalCommittedBytes,gc=new[]{GC.CollectionCount(0),GC.CollectionCount(1),GC.CollectionCount(2)}}; }
static void Write(string path,object value) { using var f=new FileStream(path,FileMode.CreateNew);JsonSerializer.Serialize(f,value,new JsonSerializerOptions{WriteIndented=true}); }
readonly record struct Row(int arm,string phase,int group,int position,long execute,long request,long bytes,int g0,int g1,int g2);
sealed class PrivateContext(string name,string directory):AssemblyLoadContext(name,false)
{
    protected override Assembly? Load(AssemblyName name)
    {
        if(name.Name is "Lokad.Onnx" or "Google.Protobuf" or "PairedBridge")
            return LoadFromAssemblyPath(Path.Combine(directory,name.Name+".dll"));
        if(name.Name!=null && (name.Name.StartsWith("System.",StringComparison.Ordinal)||name.Name is "System" or "netstandard" or "mscorlib"))return null;
        throw new FileNotFoundException("Unpinned private dependency: "+name.Name);
    }
}
