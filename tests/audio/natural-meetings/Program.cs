using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using Lokad.Onnx;

if(args.Length!=5)throw new ArgumentException("root artifact new-output family inputs|run");
string root=Path.GetFullPath(args[0]),artifact=Path.GetFullPath(args[1]),output=Path.GetFullPath(args[2]),family=args[3];
bool inputsOnly=args[4]=="inputs";
Require(inputsOnly||args[4]=="run","Mode");Require(family is "parakeet" or "whisper","Family");
Require(!Directory.Exists(output)&&!File.Exists(output),"Existing output");
using var process=Process.GetCurrentProcess();Require(Affinity(process)==4,"CPU2 required before runtime startup");
using var document=JsonDocument.Parse(File.ReadAllBytes(Path.Combine(artifact,"manifest.json")));var m=document.RootElement;
var flags=Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k=>k.StartsWith("LOKAD_",StringComparison.OrdinalIgnoreCase)||k.StartsWith("DOTNET_",StringComparison.OrdinalIgnoreCase)||k.StartsWith("COMPlus_",StringComparison.OrdinalIgnoreCase)).ToArray();
Require(flags.Length==0,"Unexpected runtime override");
Require(Sha(typeof(ComputationalGraph).Assembly.Location)==Text(m,"core_sha256"),"Core identity");
Require(Sha(typeof(WhisperTranscriber).Assembly.Location)==Text(m,"data_sha256"),"Data identity");
var model=m.GetProperty("families").GetProperty(family);
foreach(var file in model.GetProperty("models").EnumerateObject())Verify(file.Value);
string models=Path.Combine(root,Text(model,"model_directory"));
var cases=m.GetProperty("cases").EnumerateArray().Select(c=>
{
    string path=Verify(c.GetProperty("audio"));using var stream=File.OpenRead(path);
    var (all,rate)=WaveAudio.ReadMono(stream,TimeSpan.FromSeconds(600));int count=c.GetProperty("samples").GetInt32();
    Require(rate==16000&&count>0&&count<=all.Length,"PCM layout");var pcm=count==all.Length?all:all[..count];
    Require(Hash(pcm)==Text(c,"pcm_sha256"),"Decoded PCM identity");return new Input(Text(c,"name"),pcm,Hash(pcm));
}).ToArray();
Require(cases.Select(c=>c.Name).SequenceEqual(new[]{"ES2004a","IS1009a","ES2004a-recovery30"}),"Case coverage");
var json=new JsonSerializerOptions{WriteIndented=true,PropertyNamingPolicy=JsonNamingPolicy.SnakeCaseLower,Converters={new JsonStringEnumConverter()}};
Directory.CreateDirectory(output);NoOrt();
if(inputsOnly)
{
    Write("inputs.json",new{passed=true,family,cases=cases.Select(c=>new{name=c.Name,samples=c.Pcm.Length,pcm_sha256=c.Hash}),affinity=Affinity(process)});return 0;
}
long setup=Stopwatch.GetTimestamp();
ParakeetTranscriber? parakeet=family=="parakeet"?new(models):null;
WhisperTranscriber? whisper=family=="whisper"?new(models):null;
double setupSeconds=Stopwatch.GetElapsedTime(setup).TotalSeconds;
var held=new List<(object Result,string Snapshot)>();var records=new List<object>();
for(int i=0;i<cases.Length;i++)
{
    var c=cases[i];Require(Hash(c.Pcm)==c.Hash,"Input changed before call");
    foreach(var item in held)Require(JsonSerializer.Serialize(item.Result,json)==item.Snapshot,"Held result changed before call");
    int[] before=Enumerable.Range(0,3).Select(GC.CollectionCount).ToArray();long allocation=GC.GetTotalAllocatedBytes();
    long start=Stopwatch.GetTimestamp();object actual=parakeet is not null
        ?parakeet.TranscribeRecording(c.Pcm,16000,ParakeetRecordingOptions.Default,CancellationToken.None)
        :whisper!.TranscribeRecording(c.Pcm,16000,WhisperRecordingOptions.ForLanguage("en"),CancellationToken.None);
    long end=Stopwatch.GetTimestamp();long bytes=GC.GetTotalAllocatedBytes()-allocation;int[] after=Enumerable.Range(0,3).Select(GC.CollectionCount).ToArray();
    held.Add((actual,JsonSerializer.Serialize(actual,json)));Require(Hash(c.Pcm)==c.Hash,"Input changed after call");NoOrt();
    var row=new{name=c.Name,seconds=(end-start)/(double)Stopwatch.Frequency,start_ticks=start,end_ticks=end,frequency=Stopwatch.Frequency,
        result=actual,input_sha256=c.Hash,ownership=true,allocated_bytes=bytes,gc_before=before,gc_after=after};
    records.Add(row);Write($"{i:D2}.json",row);Console.WriteLine($"Complete {family} {c.Name}: {row.seconds:F3}s");
}
foreach(var item in held)Require(JsonSerializer.Serialize(item.Result,json)==item.Snapshot,"Final held result changed");
Require(cases.All(c=>Hash(c.Pcm)==c.Hash),"Final input changed");
Write("result.json",new{schema=1,engine="managed",family,records,setup_seconds=setupSeconds,held_outputs_unchanged=true,
    manifest_sha256=Sha(Path.Combine(artifact,"manifest.json")),core_sha256=Sha(typeof(ComputationalGraph).Assembly.Location),data_sha256=Sha(typeof(WhisperTranscriber).Assembly.Location),
    runner_sha256=Sha(Assembly.GetExecutingAssembly().Location),runtime=RuntimeInformation.FrameworkDescription,affinity=Affinity(process),flags});return 0;

string Verify(JsonElement value)
{
    string path=Path.Combine(root,Text(value,"path"));Require(new FileInfo(path).Length==value.GetProperty("bytes").GetInt64()&&Sha(path)==Text(value,"sha256"),"File identity: "+path);return path;
}
void Write(string name,object value){using var stream=new FileStream(Path.Combine(output,name),FileMode.CreateNew);JsonSerializer.Serialize(stream,value,json);}
static string Text(JsonElement e,string key)=>e.GetProperty(key).GetString()??throw new InvalidDataException(key);
static void Require(bool value,string message){if(!value)throw new InvalidDataException(message);}
static long Affinity(Process p)
{
    if(OperatingSystem.IsWindows()||OperatingSystem.IsLinux())return p.ProcessorAffinity.ToInt64();throw new PlatformNotSupportedException();
}
static string Sha(string p){using var stream=File.OpenRead(p);return Convert.ToHexStringLower(SHA256.HashData(stream));}
static string Hash(float[] values)=>Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
static void NoOrt(){using var p=Process.GetCurrentProcess();Require(!p.Modules.Cast<ProcessModule>().Any(m=>m.ModuleName.Contains("onnxruntime",StringComparison.OrdinalIgnoreCase)),"ORT loaded in managed process");}
sealed record Input(string Name,float[] Pcm,string Hash);
