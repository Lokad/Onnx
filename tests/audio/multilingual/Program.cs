using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

if (args.Length != 3) throw new ArgumentException("root manifest.json new-output-directory");
string root=Path.GetFullPath(args[0]), manifestPath=Path.GetFullPath(args[1]), output=Path.GetFullPath(args[2]);
Require(!Directory.Exists(output),"Output exists");
var options=new JsonSerializerOptions { WriteIndented=true, Converters={new JsonStringEnumConverter()} };
var flags=Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k=>k.StartsWith("LOKAD_",StringComparison.OrdinalIgnoreCase)
    || k.StartsWith("DOTNET_",StringComparison.OrdinalIgnoreCase)||k.StartsWith("COMPlus_",StringComparison.OrdinalIgnoreCase)).ToArray();
Require(flags.Length==0 && Environment.ProcessorCount==1 && Affinity()==4,"Normal runtime on inherited CPU2 required");
Require(Sha(typeof(ComputationalGraph).Assembly.Location)=="187de61ad8f034b9b7ad2fb3490358443fa84334204720e81bc3546a31f3c8d4","Core identity");
Require(Sha(typeof(WhisperTranscriber).Assembly.Location)=="809242b58725c6ae47514cc3908ef59ffafae6be36bb6e2fba20144d9a975af5","Data identity");
using var manifest=JsonDocument.Parse(File.ReadAllBytes(manifestPath));var m=manifest.RootElement;
string family=m.GetProperty("family").GetString()!;Require(family is "parakeet" or "whisper","Family");
foreach(var file in m.GetProperty("models").EnumerateObject())Verify(file.Value);
string audioPath=Verify(m.GetProperty("audio"));using var audio=JsonDocument.Parse(File.ReadAllBytes(audioPath));
Require(audio.RootElement.GetProperty("protocol").GetString()=="multilingual-noise-asr-v2","Protocol");
var cases=audio.RootElement.GetProperty("cases").EnumerateArray().ToArray();
Require(cases.Length==40 && cases.Select(c=>c.GetProperty("name").GetString()).Distinct().Count()==40,"Case coverage");
string models=Path.Combine(root,m.GetProperty("model_directory").GetString()!);
Directory.CreateDirectory(output);NoOrt();
var watch=Stopwatch.StartNew();
ParakeetTranscriber? parakeet=family=="parakeet" ? new(models) : null;
WhisperTranscriber? whisper=family=="whisper" ? new(models) : null;
double constructorSeconds=watch.Elapsed.TotalSeconds;
var rows=new List<object>();var held=new List<(object Result,string Json,float[] Pcm,byte[] Original)>();
for(int index=0;index<=40;index++)
{
    var c=cases[index%40];string name=c.GetProperty("name").GetString()!,language=c.GetProperty("language").GetString()!;
    Require(language is "en" or "fr" or "de" or "es" or "it","Language policy");
    string pcmPath=Path.Combine(Path.GetDirectoryName(audioPath)!,c.GetProperty("pcm").GetString()!);
    Require(Sha(pcmPath)==c.GetProperty("pcm_sha256").GetString(),"PCM identity");
    var (pcm,shape)=NpySupport.ReadFloat32(pcmPath);
    Require(shape.SequenceEqual(new[]{c.GetProperty("samples").GetInt32()}) && pcm.Length<=480000
        && pcm.All(float.IsFinite) && pcm.Any(v=>v!=0) && pcm.All(v=>Math.Abs(v)<=1),"PCM geometry/values");
    byte[] original=MemoryMarshal.AsBytes(pcm.AsSpan()).ToArray();object result,decision;
    long start=Stopwatch.GetTimestamp();
    if(parakeet is not null)
    {
        var actual=parakeet.Transcribe(pcm,16000,ParakeetTranscriptionOptions.Default,CancellationToken.None);
        long end=Stopwatch.GetTimestamp();result=actual;
        decision=new {text=actual.Text,token_ids=actual.TokenIds,frame_indices=actual.FrameIndices,duration_frames=actual.DurationFrames,
            stop_reason=actual.StopReason.ToString(),encoded_frames=actual.EncodedFrames,decoder_calls=actual.DecoderCalls};
        Record(end);
    }
    else
    {
        var actual=whisper!.Transcribe(pcm,16000,WhisperTranscriptionOptions.ForLanguage(language),CancellationToken.None);
        long end=Stopwatch.GetTimestamp();result=actual;
        decision=new {text=actual.Text,token_ids=actual.TokenIds,stop_reason=actual.StopReason.ToString(),skipped_as_no_speech=actual.SkippedAsNoSpeech};
        Record(end);
    }
    void Record(long end)
    {
        held.Add((result,JsonSerializer.Serialize(result,options),pcm,original));
        foreach(var item in held)Require(MemoryMarshal.AsBytes(item.Pcm.AsSpan()).SequenceEqual(item.Original)
            && JsonSerializer.Serialize(item.Result,options)==item.Json,"Input or held output changed");
        NoOrt();
        var row=new {name,language,repeat=index==40,decision,result,seconds=(end-start)/(double)Stopwatch.Frequency,
            start_ticks=start,end_ticks=end,frequency=Stopwatch.Frequency,
            pcm_sha256=Sha(pcmPath),input_and_held_results_unchanged=true};
        rows.Add(row);Write(Path.Combine(output,$"{index:D2}.json"),row);
        Console.WriteLine($"{index:D2} {name} language={language} seconds={row.seconds:F3}");
    }
}
Write(Path.Combine(output,"result.json"),new {schema=1,protocol="multilingual-noise-asr-v2",family,engine="managed",passed=true,
    manifest_sha256=Sha(manifestPath),audio_sha256=Sha(audioPath),core_sha256=Sha(typeof(ComputationalGraph).Assembly.Location),
    data_sha256=Sha(typeof(WhisperTranscriber).Assembly.Location),runner_sha256=Sha(Assembly.GetExecutingAssembly().Location),
    runtime=RuntimeInformation.FrameworkDescription,affinity=Affinity(),flags,constructor_seconds=constructorSeconds,cases=rows});

string Verify(JsonElement file)
{
    string path=Path.Combine(root,file.GetProperty("path").GetString()!);
    Require(new FileInfo(path).Length==file.GetProperty("bytes").GetInt64() && Sha(path)==file.GetProperty("sha256").GetString(),"File identity: "+path);return path;
}
static void Require(bool value,string message) { if(!value)throw new InvalidDataException(message); }
static string Sha(string path) { using var f=File.OpenRead(path);return Convert.ToHexStringLower(SHA256.HashData(f)); }
void Write(string path,object value) { using var f=new FileStream(path,FileMode.CreateNew);JsonSerializer.Serialize(f,value,options); }
static long Affinity()
{
    if(OperatingSystem.IsWindows() || OperatingSystem.IsLinux())return Process.GetCurrentProcess().ProcessorAffinity.ToInt64();
    throw new PlatformNotSupportedException();
}
static void NoOrt() => Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m=>m.ModuleName.Contains("onnxruntime",StringComparison.OrdinalIgnoreCase)),"Native ORT loaded");
