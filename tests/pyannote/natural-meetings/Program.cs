using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

if(args.Length!=4)throw new ArgumentException("root artifact new-output inputs|run");
string root=Path.GetFullPath(args[0]), artifact=Path.GetFullPath(args[1]), output=Path.GetFullPath(args[2]);
bool inputsOnly=args[3]=="inputs";
Require(inputsOnly || args[3]=="run","Mode");
Require(!Directory.Exists(output) && !File.Exists(output),"Existing output");
Require(OperatingSystem.IsWindows() || OperatingSystem.IsLinux(),"Platform");
using var process=Process.GetCurrentProcess();
Require(Affinity(process)==4,"CPU2 required before runtime startup");
using var document=JsonDocument.Parse(File.ReadAllBytes(Path.Combine(artifact,"manifest.json")));
var manifest=document.RootElement;
var flags=Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k=>k.StartsWith("LOKAD_",StringComparison.OrdinalIgnoreCase)||k.StartsWith("COMPlus_",StringComparison.OrdinalIgnoreCase)||k.StartsWith("DOTNET_",StringComparison.OrdinalIgnoreCase)).ToArray();
Require(flags.Length==0,"Unexpected execution override");
Require(Sha(typeof(ComputationalGraph).Assembly.Location)==Text(manifest,"core_sha256"),"Core identity");
Require(Sha(typeof(Community1Diarizer).Assembly.Location)==Text(manifest,"data_sha256"),"Data identity");
var models=manifest.GetProperty("models").EnumerateObject().ToDictionary(p=>p.Name,p=>
{
    string path=Path.Combine(root,Text(p.Value,OperatingSystem.IsLinux()?"amd_path":"path"));
    Require(new FileInfo(path).Length==p.Value.GetProperty("bytes").GetInt64() && Sha(path)==Text(p.Value,"sha256"),"Model "+p.Name);
    return path;
});
var cases=manifest.GetProperty("cases").EnumerateArray().Select(c=>
{
    string path=Path.Combine(artifact,"inputs",Text(c,"path"));
    Require(Sha(path)==Text(c.GetProperty("wav"),"sha256") && new FileInfo(path).Length==c.GetProperty("wav").GetProperty("bytes").GetInt64(),"WAV identity");
    using var stream=File.OpenRead(path);
    var (all,rate)=WaveAudio.ReadMono(stream,TimeSpan.FromSeconds(600));
    int count=c.GetProperty("samples").GetInt32();
    Require(rate==16000 && count>0 && count<=all.Length,"PCM layout");
    var pcm=count==all.Length?all:all[..count];
    Require(Hash(pcm)==Text(c,"pcm_sha256"),"Decoded PCM identity");
    return new Input(Text(c,"name"),pcm,Hash(pcm));
}).ToArray();
Require(cases.Select(c=>c.Name).SequenceEqual(new[]{"ES2004a","IS1009a","ES2004a-recovery30"}),"Case coverage");
NoOrt();Directory.CreateDirectory(output);
var json=new JsonSerializerOptions{WriteIndented=true};
if(inputsOnly)
{
    Write("inputs.json",new{passed=true,cases=cases.Select(c=>new{name=c.Name,samples=c.Pcm.Length,pcm_sha256=c.Hash}),affinity=Affinity(process)});
    return 0;
}
long setup=Stopwatch.GetTimestamp();
var model=new Community1Diarizer(models["segmentation"],models["encoder"],models["projection"],models["plda"]);
double setupSeconds=Stopwatch.GetElapsedTime(setup).TotalSeconds;
var held=new List<(Community1Diarization Result,string Snapshot)>();var records=new List<object>();
for(int i=0;i<cases.Length;i++)
{
    var c=cases[i];Require(Hash(c.Pcm)==c.Hash,"Input changed before call");
    foreach(var item in held)Require(JsonSerializer.Serialize(item.Result)==item.Snapshot,"Held result changed before call");
    int[] before=Enumerable.Range(0,3).Select(GC.CollectionCount).ToArray();long allocation=GC.GetTotalAllocatedBytes();
    long start=Stopwatch.GetTimestamp();
    var actual=model.Diarize(c.Pcm,16000,CancellationToken.None);
    long end=Stopwatch.GetTimestamp();long bytes=GC.GetTotalAllocatedBytes()-allocation;
    int[] after=Enumerable.Range(0,3).Select(GC.CollectionCount).ToArray();
    var normalized=new{status=actual.Status.ToString(),windows=actual.Windows,audio_seconds=actual.AudioDuration,
        intervals=actual.Intervals.Select(v=>new[]{v.Start,v.End,(double)v.Speaker}),
        exclusive_intervals=actual.ExclusiveIntervals.Select(v=>new[]{v.Start,v.End,(double)v.Speaker}),
        speakers=actual.Speakers.Select(s=>new{speaker=s.Speaker,centroid=s.Centroid,has_embedding=s.HasEmbedding})};
    held.Add((actual,JsonSerializer.Serialize(actual)));
    Require(Hash(c.Pcm)==c.Hash,"Input changed after call");NoOrt();
    var row=new{name=c.Name,seconds=(end-start)/(double)Stopwatch.Frequency,start_ticks=start,end_ticks=end,frequency=Stopwatch.Frequency,
        result=normalized,input_sha256=c.Hash,ownership=true,allocated_bytes=bytes,gc_before=before,gc_after=after};
    records.Add(row);Write($"{i:D2}.json",row);Console.WriteLine($"Complete {c.Name}: {row.seconds:F3}s");
}
foreach(var item in held)Require(JsonSerializer.Serialize(item.Result)==item.Snapshot,"Final held result changed");
Require(cases.All(c=>Hash(c.Pcm)==c.Hash),"Final input changed");
Write("result.json",new{schema=1,engine="managed",records,setup_seconds=setupSeconds,held_outputs_unchanged=true,
    manifest_sha256=Sha(Path.Combine(artifact,"manifest.json")),core_sha256=Sha(typeof(ComputationalGraph).Assembly.Location),data_sha256=Sha(typeof(Community1Diarizer).Assembly.Location),
    runner_sha256=Sha(Assembly.GetExecutingAssembly().Location),runtime=RuntimeInformation.FrameworkDescription,affinity=Affinity(process),flags});
return 0;

void Write(string name,object value){using var stream=new FileStream(Path.Combine(output,name),FileMode.CreateNew);JsonSerializer.Serialize(stream,value,json);}
static string Text(JsonElement e,string key)=>e.GetProperty(key).GetString()??throw new InvalidDataException(key);
static void Require(bool value,string message){if(!value)throw new InvalidDataException(message);}
static long Affinity(Process p)
{
    if(OperatingSystem.IsWindows() || OperatingSystem.IsLinux())return p.ProcessorAffinity.ToInt64();
    throw new PlatformNotSupportedException();
}
static string Sha(string p){using var stream=File.OpenRead(p);return Convert.ToHexStringLower(SHA256.HashData(stream));}
static string Hash(float[] values)=>Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
static void NoOrt(){using var p=Process.GetCurrentProcess();Require(!p.Modules.Cast<ProcessModule>().Any(m=>m.ModuleName.Contains("onnxruntime",StringComparison.OrdinalIgnoreCase)),"ORT loaded into managed process");}
sealed record Input(string Name,float[] Pcm,string Hash);
