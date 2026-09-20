using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text.Json;

if(args.Length!=4)throw new ArgumentException("Expected binary directory, frozen input artifact, new output, corpus|dc-check.");
string binaries=Path.GetFullPath(args[0]), inputs=Path.GetFullPath(args[1]), output=Path.GetFullPath(args[2]);
if(Directory.Exists(output))throw new IOException("Output exists.");Directory.CreateDirectory(output);
static string Hash(ReadOnlySpan<byte> v)=>Convert.ToHexStringLower(SHA256.HashData(v));
static object Pin(string path) { using var f=File.OpenRead(path);return new {bytes=f.Length,sha256=Convert.ToHexStringLower(SHA256.HashData(f))}; }
var context=new AssemblyLoadContext("PublicFrontend",isCollectible:true);
context.Resolving+=(_,name)=>File.Exists(Path.Combine(binaries,name.Name+".dll"))?context.LoadFromAssemblyPath(Path.Combine(binaries,name.Name+".dll")):null;
var assembly=context.LoadFromAssemblyPath(Path.Combine(binaries,"Lokad.Onnx.Data.dll"));
var type=assembly.GetType("Lokad.Onnx.WeSpeakerAudio",true)!;
var frontend=type.GetMethod("LogMelFilterbank",BindingFlags.Public|BindingFlags.Static)!.CreateDelegate<Frontend>();
static Memory<float> Buffer(object result)=>(Memory<float>)result.GetType().GetProperty("Buffer")!.GetValue(result)!;
var rows=new List<object>();var held=new List<(object Tensor,string Hash)>();
if(args[3]=="corpus")
{
    using var document=JsonDocument.Parse(File.ReadAllText(Path.Combine(inputs,"manifest.json")));
    foreach(var item in document.RootElement.GetProperty("cases").EnumerateArray())
    {
        string name=item.GetProperty("name").GetString()!;
        float[] pcm=MemoryMarshal.Cast<byte,float>(File.ReadAllBytes(Path.Combine(inputs,"inputs",name+".f32"))).ToArray();
        string before=Hash(MemoryMarshal.AsBytes(pcm.AsSpan()));var tensor=frontend(pcm,16000,CancellationToken.None);
        var dimensions=tensor.GetType().GetProperty("Dimensions")!.GetMethod!.CreateDelegate<Dimensions>(tensor)().ToArray();
        var values=Buffer(tensor);if(!dimensions.SequenceEqual(new[]{1,1+(pcm.Length-400)/160,80})||values.Length!=dimensions[1]*80)throw new InvalidDataException("Shape");
        if(Hash(MemoryMarshal.AsBytes(pcm.AsSpan()))!=before)throw new InvalidDataException("Input changed");
        string path=Path.Combine(output,name+".f32");using(var f=new FileStream(path,FileMode.CreateNew))f.Write(MemoryMarshal.AsBytes(values.Span));
        held.Add((tensor,Hash(MemoryMarshal.AsBytes(values.Span))));rows.Add(new {name,shape=dimensions,input=before,output=Pin(path)});
        Console.WriteLine(name);
    }
    foreach(var saved in held)if(Hash(MemoryMarshal.AsBytes(Buffer(saved.Tensor).Span))!=saved.Hash)throw new InvalidDataException("Held result changed");
}
else if(args[3]=="dc-check")
{
    foreach(int length in new[]{560,16000})
    {
        uint state=20260920;var pcm=new float[length];
        for(int i=0;i<length;i++){state=unchecked(state*1664525+1013904223);pcm[i]=((int)(state>>27)-16)/1048576f;}
        var shifted=pcm.Select(v=>v+.5f).ToArray();if(!pcm.SequenceEqual(shifted.Select(v=>v-.5f)))throw new InvalidDataException("Inexact DC shift");
        var expected=Buffer(frontend(pcm,16000,CancellationToken.None)).ToArray();var actual=Buffer(frontend(shifted,16000,CancellationToken.None)).ToArray();
        int failed=0;double maximum=0;
        for(int i=0;i<actual.Length;i++){double error=Math.Abs((double)actual[i]-expected[i])/Math.Max(1,Math.Abs((double)expected[i]));if(error>1e-4)failed++;maximum=Math.Max(maximum,error);}
        rows.Add(new {length,values=actual.Length,failed,maximum});
    }
}
else throw new ArgumentException("Unknown phase");
var tables=new Dictionary<string,object>();
foreach(string name in new[]{"Window","MelWeights"})
{
    var values=(float[])type.GetField(name,BindingFlags.NonPublic|BindingFlags.Static)!.GetValue(null)!;string path=Path.Combine(output,name+".f32");
    using(var f=new FileStream(path,FileMode.CreateNew))f.Write(MemoryMarshal.AsBytes(values.AsSpan()));tables.Add(name,Pin(path));
}
using var process=Process.GetCurrentProcess();
if(!OperatingSystem.IsWindows()&&!OperatingSystem.IsLinux())throw new PlatformNotSupportedException();
using(var f=new FileStream(Path.Combine(output,"result.json"),FileMode.CreateNew))JsonSerializer.Serialize(f,new {complete=true,phase=args[3],rows,tables,
    runtime=new {pid=process.Id,framework=Environment.Version.ToString(),affinity=process.ProcessorAffinity.ToInt64(),processor_count=Environment.ProcessorCount},
    loaded=context.Assemblies.Select(a=>new {name=a.GetName().Name,file=a.Location,pin=Pin(a.Location)}).ToArray()},new JsonSerializerOptions{WriteIndented=true});
delegate object Frontend(ReadOnlySpan<float> pcm,int sampleRate,CancellationToken cancellation);
delegate ReadOnlySpan<int> Dimensions();
