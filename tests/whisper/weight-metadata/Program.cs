using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

if(args.Length!=3)throw new ArgumentException("models unshared|shared output");
if(!OperatingSystem.IsWindows()&&!OperatingSystem.IsLinux())throw new PlatformNotSupportedException();
using var process=Process.GetCurrentProcess();Require(process.ProcessorAffinity.ToInt64()==4,"CPU2 before CLR");
string output=args[2];if(Directory.Exists(output))throw new IOException("Existing output");Directory.CreateDirectory(output);
Require(args[1] is "shared" or "unshared","Mode");
var first=OnnxImport.Load(Path.Combine(args[0],"decoder_model.onnx"),64L*1024*1024)!;
var past=OnnxImport.Load(Path.Combine(args[0],"decoder_with_past_model.onnx"),64L*1024*1024)!;
long shared=0;
if(args[1]=="shared")shared=(long)typeof(WhisperTranscriber).Assembly.GetType("Lokad.Onnx.WhisperDecoderWeights",true)!
    .GetMethod("Share",BindingFlags.Static|BindingFlags.NonPublic)!.Invoke(null,new object[]{first,past})!;
Require(shared==(args[1]=="shared"?635187200:0),"Shared payload");
var firstExecution=first.CreateExecution(ExecutionOptions.Memory,128L*1024*1024);
var pastExecution=past.CreateExecution(ExecutionOptions.Memory,128L*1024*1024);
var options=new JsonSerializerOptions{WriteIndented=true};var snapshots=new List<object>();var calls=new List<object>();
var held=new List<(ITensor Value,string Digest)>();var hidden=new DenseTensor<float>(new float[1500*1280],new[]{1,1500,1280});
string hiddenDigest=Digest(hidden);
void SnapshotToFile(string stage)
{
    var physical=Physical(first,past);var value=new{stage,first=Snapshot(first),past=Snapshot(past),unique_arrays=physical.Count,unique_payload_bytes=physical.Bytes};
    File.WriteAllText(Path.Combine(output,$"snapshot-{snapshots.Count:D2}.json"),JsonSerializer.Serialize(value,options));snapshots.Add(value);
}
Dictionary<string,ITensor> Execute(GraphExecution context,Dictionary<string,ITensor> feeds,string name)
{
    var inputs=feeds.ToDictionary(p=>p.Key,p=>Digest(p.Value));context.Reset();
    Require(context.Execute(feeds,true,ExecutionProvider.CPU,ExecutionOptions.Memory),context.LastErrorMessage??"Execute failed");
    var result=context.Outputs.ToDictionary(p=>p.Key,p=>p.Value!);
    foreach(var entry in feeds)Require(Digest(entry.Value)==inputs[entry.Key],"Feed mutated");
    foreach(var entry in result)held.Add((entry.Value,Digest(entry.Value)));
    foreach(var old in held)Require(Digest(old.Value)==old.Digest,"Held output mutated");
    Require(Digest(hidden)==hiddenDigest,"Hidden input mutated");
    var call=new{name,outputs=result.OrderBy(p=>p.Key,StringComparer.Ordinal).Select(p=>new{name=p.Key,shape=p.Value.Dims,type=p.Value.ElementType.ToString(),sha256=Digest(p.Value)}).ToArray()};
    File.WriteAllText(Path.Combine(output,$"call-{calls.Count:D2}.json"),JsonSerializer.Serialize(call,options));calls.Add(call);
    context.Reset();return result;
}
SnapshotToFile("before");
for(int repeat=0;repeat<2;repeat++)
{
    var firstResult=Execute(firstExecution,new Dictionary<string,ITensor>{["input_ids"]=new DenseTensor<long>(new long[]{50258,50259,50359,50363},new[]{1,4}),["encoder_hidden_states"]=hidden},$"first-{repeat}");
    SnapshotToFile($"after-first-{repeat}");
    var feeds=new Dictionary<string,ITensor>{["input_ids"]=new DenseTensor<long>(new long[]{0},new[]{1,1})};
    for(int layer=0;layer<4;layer++)foreach(var attention in new[]{"decoder","encoder"})foreach(var kind in new[]{"key","value"})
    {
        string suffix=$"{layer}.{attention}.{kind}";feeds["past_key_values."+suffix]=firstResult["present."+suffix];
    }
    Execute(pastExecution,feeds,$"past-{repeat}");SnapshotToFile($"after-past-{repeat}");
}
Require(!process.Modules.Cast<ProcessModule>().Any(m=>m.ModuleName.Contains("onnxruntime",StringComparison.OrdinalIgnoreCase)),"ORT loaded");
File.WriteAllText(Path.Combine(output,"result.json"),JsonSerializer.Serialize(new{passed=true,mode=args[1],logical_shared_bytes=shared,calls,snapshots,
    held_outputs_unchanged=true,input_unchanged=true,runtime=RuntimeInformation.FrameworkDescription,affinity=process.ProcessorAffinity.ToInt64(),processor_count=Environment.ProcessorCount,
    core_sha256=Sha(typeof(ComputationalGraph).Assembly.Location),data_sha256=Sha(typeof(WhisperTranscriber).Assembly.Location),runner_sha256=Sha(Assembly.GetExecutingAssembly().Location),
    flags=Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k=>k.StartsWith("LOKAD_",StringComparison.OrdinalIgnoreCase)||k.StartsWith("DOTNET_",StringComparison.OrdinalIgnoreCase)||k.StartsWith("COMPlus_",StringComparison.OrdinalIgnoreCase)).ToDictionary(k=>k,Environment.GetEnvironmentVariable)},options));
return 0;

static void Require(bool condition,string message){if(!condition)throw new InvalidDataException(message);}
static string Sha(string path){using var stream=File.OpenRead(path);return Convert.ToHexStringLower(SHA256.HashData(stream));}
// Snapshot, Digest and Physical are appended verbatim from the closed inspection.
