using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

if (args.Length != 3) throw new ArgumentException("Usage: ParakeetRecordingReplay <models> <inputs.json> <new-output-directory>");
string models=Path.GetFullPath(args[0]), inputs=Path.GetFullPath(args[1]), destination=Path.GetFullPath(args[2]);
if (Directory.Exists(destination) || File.Exists(destination)) throw new IOException("Existing output");
static string Sha(string path) { using var f=File.OpenRead(path);return Convert.ToHexStringLower(SHA256.HashData(f)); }
static void Require(bool value,string message) { if(!value) throw new InvalidDataException(message); }
static void NoOrt()
{
    foreach(ProcessModule module in Process.GetCurrentProcess().Modules)
        Require(!module.ModuleName.Contains("onnxruntime",StringComparison.OrdinalIgnoreCase),"Native ORT loaded");
}
using var manifest=JsonDocument.Parse(File.ReadAllBytes(inputs));
using var pins=JsonDocument.Parse(Assembly.GetExecutingAssembly().GetManifestResourceStream("assets.json")!);
foreach(var file in pins.RootElement.GetProperty("files").EnumerateObject())
    Require(new FileInfo(Path.Combine(models,file.Name)).Length==file.Value.GetProperty("bytes").GetInt64()
        && Sha(Path.Combine(models,file.Name))==file.Value.GetProperty("sha256").GetString(),"Model hash differs: "+file.Name);
var cases=manifest.RootElement.GetProperty("cases").EnumerateArray().ToArray();
Require(manifest.RootElement.GetProperty("schema").GetInt32()==1 && manifest.RootElement.GetProperty("sample_rate").GetInt32()==16000
    && cases.Select(c=>c.GetProperty("name").GetString()).SequenceEqual(new[]{"connected","shifted","hard-boundary","token-limit","window-limit","maximum-speech","tiny-tail","maximum-silence","empty"}),"Coverage differs");
var json=new JsonSerializerOptions { WriteIndented=true,PropertyNamingPolicy=JsonNamingPolicy.SnakeCaseLower,Converters={new JsonStringEnumConverter()} };
Directory.CreateDirectory(destination);NoOrt();
var watch=Stopwatch.StartNew();var model=new ParakeetTranscriber(models);double load=watch.Elapsed.TotalSeconds;
var policy=ParakeetRecordingOptions.Default;var refusals=new List<string>();
void Reject<T>(string name,Action action) where T:Exception
{
    try { action(); } catch(T) { refusals.Add(name);return; }
    throw new InvalidDataException("Expected "+typeof(T).Name+": "+name);
}
Reject<ArgumentOutOfRangeException>("sample-rate",()=>model.TranscribeRecording(new float[1],8000,policy,CancellationToken.None));
Reject<ArgumentOutOfRangeException>("too-long",()=>model.TranscribeRecording(new float[9600001],16000,policy,CancellationToken.None));
foreach(int limit in new[]{0,513}) Reject<ArgumentOutOfRangeException>("windows-"+limit,()=>model.TranscribeRecording(Array.Empty<float>(),16000,policy with{MaxWindows=limit},CancellationToken.None));
foreach(int limit in new[]{0,4097}) Reject<ArgumentOutOfRangeException>("tokens-"+limit,()=>model.TranscribeRecording(Array.Empty<float>(),16000,policy with{Decoding=policy.Decoding with{MaxTokens=limit}},CancellationToken.None));
foreach(int limit in new[]{0,11}) Reject<ArgumentOutOfRangeException>("per-frame-"+limit,()=>model.TranscribeRecording(Array.Empty<float>(),16000,policy with{Decoding=policy.Decoding with{MaxTokensPerFrame=limit}},CancellationToken.None));
foreach(float value in new[]{float.NaN,float.PositiveInfinity,float.NegativeInfinity}) Reject<ArgumentException>("nonfinite-"+value,()=>model.TranscribeRecording(new[]{value},16000,policy,CancellationToken.None));
Reject<ArgumentNullException>("null-options",()=>model.TranscribeRecording(Array.Empty<float>(),16000,null!,CancellationToken.None));
Reject<ArgumentNullException>("null-decoding",()=>model.TranscribeRecording(Array.Empty<float>(),16000,policy with{Decoding=null!},CancellationToken.None));
Reject<ArgumentOutOfRangeException>("short-api-bound",()=>model.Transcribe(new float[480001],16000,policy.Decoding,CancellationToken.None));
using(var canceled=new CancellationTokenSource())
{ canceled.Cancel();Reject<OperationCanceledException>("pre-canceled",()=>model.TranscribeRecording(Array.Empty<float>(),16000,policy,canceled.Token)); }
float[] Pcm(JsonElement item)
{
    string path=Path.Combine(Path.GetDirectoryName(inputs)!,item.GetProperty("pcm").GetString()!);
    Require(Sha(path)==item.GetProperty("pcm_sha256").GetString(),"PCM hash differs");
    var (pcm,shape)=NpySupport.ReadFloat32(path);
    Require(shape.SequenceEqual(new[]{item.GetProperty("samples").GetInt32()}) && pcm.All(float.IsFinite),"PCM contract differs");return pcm;
}
using(var cancellation=new CancellationTokenSource())
{
    float[] pcm=Pcm(cases[0]);cancellation.CancelAfter(50);
    Reject<OperationCanceledException>("during-inference",()=>model.TranscribeRecording(pcm,16000,policy,cancellation.Token));
}
var held=new List<(ParakeetRecording Result,string Json,float[] Pcm,byte[] Original)>();var rows=new List<object>();
for(int i=0;i<=cases.Length;i++)
{
    var item=cases[i%cases.Length];string name=item.GetProperty("name").GetString()!;float[] pcm=Pcm(item);
    byte[] original=MemoryMarshal.AsBytes(pcm.AsSpan()).ToArray();
    var options=new ParakeetRecordingOptions(new(item.GetProperty("max_tokens").GetInt32(),item.GetProperty("max_tokens_per_frame").GetInt32()),item.GetProperty("max_windows").GetInt32());
    watch.Restart();var result=model.TranscribeRecording(pcm,16000,options,CancellationToken.None);double seconds=watch.Elapsed.TotalSeconds;
    string serialized=JsonSerializer.Serialize(result,json);held.Add((result,serialized,pcm,original));
    foreach(var prior in held) Require(JsonSerializer.Serialize(prior.Result,json)==prior.Json && MemoryMarshal.AsBytes(prior.Pcm.AsSpan()).SequenceEqual(prior.Original),"Held result/input changed");
    rows.Add(new{name,repeat=i==cases.Length,result,seconds});
    File.WriteAllText(Path.Combine(destination,"progress.json"),JsonSerializer.Serialize(rows,json));
    Console.WriteLine($"{i:D2} {name}: {result.StopReason}, windows={result.Windows.Count}, seconds={seconds:F3}");
}
Require(held[0].Json==held[^1].Json,"Repeated recording differs");
var concurrent=await Task.WhenAll(Enumerable.Range(0,2).Select(_=>Task.Run(()=>model.TranscribeRecording(new float[31*16000],16000,policy,CancellationToken.None))));
Require(concurrent.All(r=>r.StopReason==ParakeetRecordingStopReason.Completed && r.Windows.Count==2 && r.Text=="" && r.ProcessedSeconds==31),"Concurrent silent requests differ");
NoOrt();
File.WriteAllText(Path.Combine(destination,"result.json"),JsonSerializer.Serialize(new{schema=1,inputs_sha256=Sha(inputs),cases=rows,refusals,concurrent,ownership=true,load_seconds=load,
    core_sha256=Sha(typeof(ComputationalGraph).Assembly.Location),data_sha256=Sha(typeof(ParakeetTranscriber).Assembly.Location),runner_sha256=Sha(Assembly.GetExecutingAssembly().Location),
    runtime=RuntimeInformation.FrameworkDescription,os=RuntimeInformation.OSDescription,peak_working_set=Process.GetCurrentProcess().PeakWorkingSet64,
    flags=Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k=>k.StartsWith("LOKAD_")||k.StartsWith("COMPlus_")||k.StartsWith("DOTNET_")).ToDictionary(k=>k,Environment.GetEnvironmentVariable)},json));
Console.WriteLine($"Complete: {rows.Count} recordings, {refusals.Count} refusals, ownership and concurrent silence pass.");
