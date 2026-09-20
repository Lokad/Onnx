using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

if(args.Length!=4)throw new ArgumentException("root manifest output conformance|timing");
if(!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux())throw new PlatformNotSupportedException("Verified process affinity required");
string root=Path.GetFullPath(args[0]),manifestPath=Path.GetFullPath(args[1]),output=Path.GetFullPath(args[2]);
bool conformance=args[3]=="conformance";Require(conformance,"Diagnostic requires conformance mode");if(!conformance && args[3]!="timing")throw new ArgumentException("mode");
if(Directory.Exists(output)||File.Exists(output))throw new IOException("Existing output");
using var process=Process.GetCurrentProcess();Require(Affinity(process)==4,"CPU2 affinity required before CLR");
using var document=JsonDocument.Parse(File.ReadAllBytes(manifestPath));var manifest=document.RootElement;
Require(manifest.GetProperty("schema").GetInt32()==1,"Schema");
string family=manifest.GetProperty("family").GetString()!;
string PathOf(JsonElement spec)
{
    string path=Path.GetFullPath(Path.Combine(root,spec.GetProperty("path").GetString()!));
    Require(Sha(path)==spec.GetProperty("sha256").GetString() && new FileInfo(path).Length==spec.GetProperty("bytes").GetInt64(),"File identity: "+path);return path;
}
var models=manifest.GetProperty("models").EnumerateObject().ToDictionary(p=>p.Name,p=>PathOf(p.Value));
PathOf(manifest.GetProperty("reference"));
var cases=manifest.GetProperty("cases").EnumerateArray().Select(c=>
{
    string path=PathOf(c.GetProperty("pcm"));var (pcm,shape)=NpySupport.ReadFloat32(path);
    Require(shape.SequenceEqual(new[]{c.GetProperty("samples").GetInt32()}) && pcm.All(float.IsFinite),"PCM layout");
    return new Case(c.GetProperty("name").GetString()!,pcm,Hash(pcm),c.GetProperty("expected").Clone());
}).ToArray();
Require(cases.Length==20,"Diagnostic requires complete twenty-case corpus");
var collections=new List<object>();
var json=new JsonSerializerOptions{WriteIndented=true};Directory.CreateDirectory(output);NoOrt();
long setupStart=Stopwatch.GetTimestamp();
Require(family=="whisper","Unknown family");
var whisper=new WhisperTranscriber(Path.GetDirectoryName(models["config.json"])!);
double setupSeconds=Stopwatch.GetElapsedTime(setupStart).TotalSeconds;
var records=new List<object>();var held=new List<(object Result,string Snapshot)>();var first=new Dictionary<string,string>();
int warmup=conformance?1:manifest.GetProperty("warmup_passes").GetInt32(),measured=conformance?0:manifest.GetProperty("measured_passes").GetInt32();
for(int pass=0;pass<warmup+measured;pass++)foreach(var c in cases)
{
    Require(Hash(c.Pcm)==c.Hash,"Input changed before request");
    foreach(var old in held)Require(JsonSerializer.Serialize(old.Result)==old.Snapshot,"Held output changed");
    var memoryBefore=Memory();
    var gcBefore=Enumerable.Range(0,3).Select(GC.CollectionCount).ToArray();long allocated=GC.GetTotalAllocatedBytes();
    long start=Stopwatch.GetTimestamp();object actual;
    actual=whisper.Transcribe(c.Pcm,16000,WhisperTranscriptionOptions.ForLanguage("en"),CancellationToken.None);
    long end=Stopwatch.GetTimestamp();long allocatedAfter=GC.GetTotalAllocatedBytes();var gcAfter=Enumerable.Range(0,3).Select(GC.CollectionCount).ToArray();var memoryAfter=Memory();
    var w=(WhisperTranscription)actual;
    JsonElement normalized=JsonSerializer.SerializeToElement(new{text=w.Text,token_ids=w.TokenIds,stop_reason=w.StopReason.ToString(),skipped_as_no_speech=w.SkippedAsNoSpeech});
    double error=Agree(normalized,c.Expected,"",family);Require(Hash(c.Pcm)==c.Hash,"Input mutation");
    string snapshot=JsonSerializer.Serialize(actual);
    if(first.TryGetValue(c.Name,out var prior))Require(prior==snapshot,"Repeat changed output");else first[c.Name]=snapshot;
    held.Add((actual,snapshot));NoOrt();
    var row=new{name=c.Name,pass,phase=pass<warmup?"warmup":"measured",seconds=(end-start)/(double)Stopwatch.Frequency,start_ticks=start,end_ticks=end,frequency=Stopwatch.Frequency,
        result=normalized,maximum_centroid_error=error,input_sha256=c.Hash,ownership=true,gc_before=gcBefore,gc_after=gcAfter,allocated_bytes=allocatedAfter-allocated,memory_before=memoryBefore,memory_after=memoryAfter};
    records.Add(row);File.WriteAllText(Path.Combine(output,$"{records.Count-1:D3}.json"),JsonSerializer.Serialize(row,json));
    Console.WriteLine($"{family} {row.phase} pass={pass} {c.Name} {row.seconds:F4}s");
    if(records.Count is 8 or 16 or 20)
    {
        var before=Memory();long collectionStart=Stopwatch.GetTimestamp();
        GC.Collect(2,GCCollectionMode.Forced,blocking:true,compacting:true);
        GC.WaitForPendingFinalizers();
        long collectionEnd=Stopwatch.GetTimestamp();var after=Memory();
        foreach(var old in held)Require(JsonSerializer.Serialize(old.Result)==old.Snapshot,"Collection changed held output");
        Require(cases.All(item=>Hash(item.Pcm)==item.Hash),"Collection changed input");
        var observation=new{after_call=records.Count,before,after,start_ticks=collectionStart,end_ticks=collectionEnd,frequency=Stopwatch.Frequency,
            seconds=(collectionEnd-collectionStart)/(double)Stopwatch.Frequency,held_outputs_unchanged=true,inputs_unchanged=true};
        collections.Add(observation);
        File.WriteAllText(Path.Combine(output,$"collection-{records.Count:D2}.json"),JsonSerializer.Serialize(observation,json));
    }
}
foreach(var old in held)Require(JsonSerializer.Serialize(old.Result)==old.Snapshot,"Final held output changed");
Require(cases.All(c=>Hash(c.Pcm)==c.Hash),"Final input mutation");
File.WriteAllText(Path.Combine(output,"result.json"),JsonSerializer.Serialize(new{schema=1,family,engine="managed",conformance,records,collections,diagnostic="explicit-gen2-after-8-16-20",setup_seconds=setupSeconds,
    manifest_sha256=Sha(manifestPath),core_sha256=Sha(typeof(ComputationalGraph).Assembly.Location),data_sha256=Sha(typeof(WhisperTranscriber).Assembly.Location),runner_sha256=Sha(Assembly.GetExecutingAssembly().Location),
    runtime=RuntimeInformation.FrameworkDescription,os=RuntimeInformation.OSDescription,affinity=Affinity(process),processor_count=Environment.ProcessorCount,
    peak_working_set=process.PeakWorkingSet64,held_outputs_unchanged=true,flags=Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k=>k.StartsWith("LOKAD_")||k.StartsWith("COMPlus_")||k.StartsWith("DOTNET_")).ToDictionary(k=>k,Environment.GetEnvironmentVariable)},json));
return 0;

static string Sha(string p){using var stream=File.OpenRead(p);return Convert.ToHexStringLower(SHA256.HashData(stream));}
static long Affinity(Process p)
{
    if(OperatingSystem.IsWindows())return p.ProcessorAffinity.ToInt64();
    if(OperatingSystem.IsLinux())return p.ProcessorAffinity.ToInt64();
    throw new PlatformNotSupportedException();
}
static string Hash(float[] a)=>Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(a.AsSpan())));
static void Require(bool ok,string message){if(!ok)throw new InvalidDataException(message);}
static void NoOrt(){using var p=Process.GetCurrentProcess();Require(!p.Modules.Cast<ProcessModule>().Any(m=>m.ModuleName.Contains("onnxruntime",StringComparison.OrdinalIgnoreCase)),"Native ORT loaded in managed worker");}
static double Agree(JsonElement actual,JsonElement expected,string path,string family)
{
    Require(actual.ValueKind==expected.ValueKind,$"Kind {path}");double maximum=0;
    switch(expected.ValueKind)
    {
        case JsonValueKind.Object:
            Require(actual.EnumerateObject().Select(p=>p.Name).Order().SequenceEqual(expected.EnumerateObject().Select(p=>p.Name).Order()),"Fields "+path);
            foreach(var p in expected.EnumerateObject())maximum=Math.Max(maximum,Agree(actual.GetProperty(p.Name),p.Value,path+"/"+p.Name,family));break;
        case JsonValueKind.Array:
            Require(actual.GetArrayLength()==expected.GetArrayLength(),"Length "+path);
            for(int i=0;i<expected.GetArrayLength();i++)maximum=Math.Max(maximum,Agree(actual[i],expected[i],path+"/"+i,family));break;
        case JsonValueKind.Number:
            double a=actual.GetDouble(),e=expected.GetDouble();Require(double.IsFinite(a)&&double.IsFinite(e),"Nonfinite "+path);
            if(path.Contains("/centroid/")){maximum=Math.Abs(a-e)/Math.Max(1,Math.Abs(e));Require(maximum<=1e-4,"Centroid "+path);}
            else Require(Math.Abs(a-e)<= (family=="pyannote" && (path.StartsWith("/intervals/")||path.StartsWith("/exclusive_intervals/"))?1e-12:0),"Value "+path);break;
        default:Require(actual.ToString()==expected.ToString(),"Value "+path);break;
    }
    return maximum;
}
static object Memory()
{
    using var process=Process.GetCurrentProcess();process.Refresh();var info=GC.GetGCMemoryInfo();
    return new{ticks=Stopwatch.GetTimestamp(),allocated_total=GC.GetTotalAllocatedBytes(true),managed_estimate=GC.GetTotalMemory(false),
        rss=process.WorkingSet64,private_bytes=process.PrivateMemorySize64,collections=Enumerable.Range(0,3).Select(GC.CollectionCount).ToArray(),
        last_gc_index=info.Index,last_gc_generation=info.Generation,last_gc_concurrent=info.Concurrent,last_gc_compacted=info.Compacted,
        last_gc_heap=info.HeapSizeBytes,last_gc_fragmented=info.FragmentedBytes,last_gc_committed=info.TotalCommittedBytes,
        last_gc_memory_load=info.MemoryLoadBytes,total_available_memory=info.TotalAvailableMemoryBytes,high_memory_threshold=info.HighMemoryLoadThresholdBytes,
        last_gc_promoted=info.PromotedBytes,last_gc_pinned=info.PinnedObjectsCount,last_gc_finalization_pending=info.FinalizationPendingCount};
}

sealed record Case(string Name,float[] Pcm,string Hash,JsonElement Expected);
