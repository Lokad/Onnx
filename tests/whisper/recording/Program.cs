using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

if (args.Length != 4) throw new ArgumentException("Usage: RecordingReplay <models> <inputs.json> <old-short-native-manifest.json> <new-output-directory>");
string models=Path.GetFullPath(args[0]), inputs=Path.GetFullPath(args[1]), shortReference=Path.GetFullPath(args[2]), destination=Path.GetFullPath(args[3]);
if (Directory.Exists(destination) || File.Exists(destination)) throw new IOException("Existing result destination");
static string Sha(string path) { using var input=File.OpenRead(path);return Convert.ToHexStringLower(SHA256.HashData(input)); }
static void Require(bool value, string message) { if (!value) throw new InvalidDataException(message); }
static void NoOrt()
{
    foreach (ProcessModule module in Process.GetCurrentProcess().Modules)
        Require(!module.ModuleName.Contains("onnxruntime",StringComparison.OrdinalIgnoreCase),"Native ORT loaded");
}
using var data=JsonDocument.Parse(File.ReadAllBytes(inputs));
using var pinStream=Assembly.GetExecutingAssembly().GetManifestResourceStream("assets.json") ?? throw new InvalidDataException("Missing asset pins");
using var pins=JsonDocument.Parse(pinStream);
foreach(var file in pins.RootElement.GetProperty("files").EnumerateObject())
    Require(new FileInfo(Path.Combine(models,file.Name)).Length==file.Value.GetProperty("bytes").GetInt64()
        && Sha(Path.Combine(models,file.Name))==file.Value.GetProperty("sha256").GetString(),"Model hash differs: "+file.Name);
var cases=data.RootElement.GetProperty("cases").EnumerateArray().ToArray();
Require(data.RootElement.GetProperty("schema").GetInt32()==1 && data.RootElement.GetProperty("sample_rate").GetInt32()==16000
    && cases.Select(c=>c.GetProperty("name").GetString()).SequenceEqual(new[]{"connected","shifted","token-limit","window-limit"}),"Input coverage differs");
var json=new JsonSerializerOptions{WriteIndented=true,PropertyNamingPolicy=JsonNamingPolicy.SnakeCaseLower,Converters={new JsonStringEnumConverter()}};
Directory.CreateDirectory(destination);NoOrt();
var watch=Stopwatch.StartNew();var model=new WhisperTranscriber(models);double load=watch.Elapsed.TotalSeconds;
var options=WhisperRecordingOptions.ForLanguage("en");
int refusals=0;
void Reject<T>(Action action) where T:Exception
{
    try { action(); } catch(T) { refusals++;return; }
    throw new InvalidDataException("Expected "+typeof(T).Name);
}
Reject<ArgumentOutOfRangeException>(()=>model.TranscribeRecording(new float[1],8000,options,CancellationToken.None));
Reject<ArgumentOutOfRangeException>(()=>model.TranscribeRecording(new float[WhisperTranscriber.MaximumRecordingSamples+1],16000,options,CancellationToken.None));
Reject<ArgumentOutOfRangeException>(()=>model.TranscribeRecording(Array.Empty<float>(),16000,options with{MaxWindows=0},CancellationToken.None));
Reject<ArgumentOutOfRangeException>(()=>model.TranscribeRecording(Array.Empty<float>(),16000,options with{MaxWindows=513},CancellationToken.None));
Reject<ArgumentException>(()=>model.TranscribeRecording(new[]{float.NaN},16000,options,CancellationToken.None));
Reject<ArgumentException>(()=>model.TranscribeRecording(new[]{float.PositiveInfinity},16000,options,CancellationToken.None));
Reject<ArgumentException>(()=>model.TranscribeRecording(Array.Empty<float>(),16000,WhisperRecordingOptions.ForLanguage("invalid"),CancellationToken.None));
Reject<ArgumentNullException>(()=>model.TranscribeRecording(Array.Empty<float>(),16000,null!,CancellationToken.None));
using(var canceled=new CancellationTokenSource())
{
    canceled.Cancel();Reject<OperationCanceledException>(()=>model.TranscribeRecording(Array.Empty<float>(),16000,options,canceled.Token));
}
var held=new List<(WhisperRecording Result,string Serialized,float[] Pcm,byte[] Original)>();
var rows=new List<object>();
float[] Pcm(JsonElement item)
{
    string path=Path.Combine(Path.GetDirectoryName(inputs)!,item.GetProperty("pcm").GetString()!);
    Require(Sha(path)==item.GetProperty("pcm_sha256").GetString(),"Input hash differs");
    var (pcm,shape)=NpySupport.ReadFloat32(path);
    Require(shape.SequenceEqual(new[]{item.GetProperty("samples").GetInt32()}) && pcm.All(float.IsFinite),"Input shape/values differ");
    return pcm;
}
using(var timed=new CancellationTokenSource())
{
    float[] pcm=Pcm(cases[0]);timed.CancelAfter(50);
    Reject<OperationCanceledException>(()=>model.TranscribeRecording(pcm,16000,options,timed.Token));
}
for(int i=0;i<=cases.Length;i++)
{
    var item=cases[i%cases.Length];string name=item.GetProperty("name").GetString()!;float[] pcm=Pcm(item);
    byte[] original=MemoryMarshal.AsBytes(pcm.AsSpan()).ToArray();
    var policy=new WhisperRecordingOptions(WhisperTranscriptionOptions.ForLanguage(item.GetProperty("language").GetString()!)
        with{MaxNewTokens=item.GetProperty("max_new_tokens").GetInt32()},item.GetProperty("max_windows").GetInt32());
    watch.Restart();var result=model.TranscribeRecording(pcm,16000,policy,CancellationToken.None);watch.Stop();
    string serialized=JsonSerializer.Serialize(result,json);held.Add((result,serialized,pcm,original));
    foreach(var prior in held)
        Require(MemoryMarshal.AsBytes(prior.Pcm.AsSpan()).SequenceEqual(prior.Original) && JsonSerializer.Serialize(prior.Result,json)==prior.Serialized,"Input or held result changed");
    if(i==cases.Length) Require(serialized==held[0].Serialized,"Repeat changed output bits/decisions");
    NoOrt();var row=new{name,repeat=i==cases.Length,result,seconds=watch.Elapsed.TotalSeconds,pcm_sha256=item.GetProperty("pcm_sha256").GetString(),ownership=true};rows.Add(row);
    File.WriteAllText(Path.Combine(destination,$"{i:D2}-{name}.json"),JsonSerializer.Serialize(row,json));
    Console.WriteLine($"{i:D2} {name}: {result.StopReason}, windows={result.Windows.Count}, segments={result.Segments.Count}, seconds={watch.Elapsed.TotalSeconds:F3}");
}
var empty=model.TranscribeRecording(Array.Empty<float>(),16000,options,CancellationToken.None);
Require(empty.StopReason==WhisperRecordingStopReason.Completed && empty.Windows.Count==0 && empty.Text=="","Empty request differs");
var silence=new float[WhisperTranscriber.MaximumRecordingSamples];
var silent=model.TranscribeRecording(silence,16000,options,CancellationToken.None);
Require(silent.StopReason==WhisperRecordingStopReason.Completed && silent.Windows.Count==20 && silent.Segments.Count==0 && silent.ProcessedSeconds==600,"Maximum silent request differs");
var concurrent=await Task.WhenAll(Enumerable.Range(0,2).Select(_=>Task.Run(()=>model.TranscribeRecording(new float[31*16000],16000,options,CancellationToken.None))));
Require(concurrent.All(r=>r.StopReason==WhisperRecordingStopReason.Completed && r.Windows.Count==2 && r.Text==""),"Concurrent silent requests differ");
using var old=JsonDocument.Parse(File.ReadAllBytes(shortReference));var oldCase=old.RootElement.GetProperty("cases")[0];
string shortPath=Path.Combine(Path.GetDirectoryName(shortReference)!,oldCase.GetProperty("pcm").GetString()!);
Require(Sha(shortPath)==oldCase.GetProperty("pcm_sha256").GetString(),"Short regression input differs");
var shortPcm=NpySupport.ReadFloat32(shortPath).Values;var shortOriginal=MemoryMarshal.AsBytes(shortPcm.AsSpan()).ToArray();
var shortResult=model.Transcribe(shortPcm,16000,WhisperTranscriptionOptions.ForLanguage("en"),CancellationToken.None);
Require(shortResult.Text==oldCase.GetProperty("text").GetString() && shortResult.TokenIds.SequenceEqual(oldCase.GetProperty("tokens").EnumerateArray().Select(t=>t.GetInt32()))
    && shortResult.StopReason.ToString()==oldCase.GetProperty("stop_reason").GetString()
    && shortResult.SkippedAsNoSpeech==oldCase.GetProperty("skipped_as_no_speech").GetBoolean(),"Existing short API application changed");
Require(MemoryMarshal.AsBytes(shortPcm.AsSpan()).SequenceEqual(shortOriginal),"Short API input changed");
foreach(var prior in held) Require(JsonSerializer.Serialize(prior.Result,json)==prior.Serialized && MemoryMarshal.AsBytes(prior.Pcm.AsSpan()).SequenceEqual(prior.Original),"Final held results/inputs changed");
NoOrt();
File.WriteAllText(Path.Combine(destination,"result.json"),JsonSerializer.Serialize(new{schema=1,inputs_sha256=Sha(inputs),cases=rows,refusals,empty,silent,concurrent,
    short_regression=shortResult,short_manifest_sha256=Sha(shortReference),ownership=true,load_seconds=load,
    core_sha256=Sha(typeof(ComputationalGraph).Assembly.Location),data_sha256=Sha(typeof(WhisperTranscriber).Assembly.Location),runner_sha256=Sha(Assembly.GetExecutingAssembly().Location),
    runtime=RuntimeInformation.FrameworkDescription,os=RuntimeInformation.OSDescription,peak_working_set=Process.GetCurrentProcess().PeakWorkingSet64,
    flags=Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k=>k.StartsWith("LOKAD_")||k.StartsWith("COMPlus_")||k.StartsWith("DOTNET_")).ToDictionary(k=>k,Environment.GetEnvironmentVariable)},json));
Console.WriteLine($"Complete: {rows.Count} recording requests, {refusals} refusals, silence/concurrency and short regression pass.");
