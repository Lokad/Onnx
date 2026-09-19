using System.Diagnostics;
using System.Reflection;
using System.Text;
using System.Text.Json.Nodes;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

string root=Path.GetFullPath(args[0]),destination=Path.GetFullPath(args[1]);
if(File.Exists(destination)||Directory.Exists(destination+".arrays"))throw new IOException("Output exists");
Directory.CreateDirectory(destination+".arrays");
string Sha(byte[] bytes)=>Convert.ToHexStringLower(SHA256.HashData(bytes));
using var doc=JsonDocument.Parse(File.ReadAllText(Path.Combine(root,"manifest.json")));var manifest=doc.RootElement;

string SourceHash(byte[] bytes) => Sha(Encoding.UTF8.GetBytes(Encoding.UTF8.GetString(bytes).Replace("\r\n","\n",StringComparison.Ordinal)));
byte[] Resource(string name) { using var stream=Assembly.GetExecutingAssembly().GetManifestResourceStream(name) ?? throw new InvalidDataException("Missing embedded reference"); using var memory=new MemoryStream(); stream.CopyTo(memory); return memory.ToArray(); }
byte[] pins=Resource("frontend-pins.json");using var pinsDocument=JsonDocument.Parse(pins);var pinned=pinsDocument.RootElement;
if(!JsonNode.DeepEquals(JsonNode.Parse(pins),JsonNode.Parse(manifest.GetProperty("pins").GetRawText()))
    || SourceHash(pins)!=manifest.GetProperty("pins_lf_sha256").GetString()
    || SourceHash(Resource("frontend-generator.py"))!=manifest.GetProperty("generator_lf_sha256").GetString()
    || manifest.GetProperty("scope").GetString()!=pinned.GetProperty("scope").GetString()
    || manifest.GetProperty("tolerance").GetDouble()!=1e-4)throw new InvalidDataException("Reference identity or gate differs");
foreach(var v in pinned.GetProperty("versions").EnumerateObject())if(manifest.GetProperty(v.Name).GetString()!=v.Value.GetString())throw new InvalidDataException("Native version differs");
if(!JsonNode.DeepEquals(JsonNode.Parse(manifest.GetProperty("settings").GetRawText()),JsonNode.Parse(pinned.GetProperty("settings").GetRawText()))
    || !manifest.GetProperty("cases").EnumerateArray().Select(c=>c.GetProperty("name").GetString()).SequenceEqual(pinned.GetProperty("cases").EnumerateArray().Select(c=>c.GetString()))
    || manifest.GetProperty("files").EnumerateObject().Count()!=107)throw new InvalidDataException("Coverage or settings differ");
var tensors=new Dictionary<string,(float[] Values,int[] Shape)>();
foreach(var f in manifest.GetProperty("files").EnumerateObject())
{
    string path=Path.GetFullPath(Path.Combine(root,f.Name));if(Path.GetDirectoryName(path)!=root||Sha(File.ReadAllBytes(path))!=f.Value.GetProperty("sha256").GetString())throw new InvalidDataException("Fixture hash");
    var array=NpySupport.ReadFloat32(path);
    if(f.Value.GetProperty("dtype").GetString()!="float32" || new FileInfo(path).Length!=f.Value.GetProperty("bytes").GetInt64()
        || !array.Shape.SequenceEqual(f.Value.GetProperty("shape").EnumerateArray().Select(x=>x.GetInt32())) || !array.Values.All(float.IsFinite))throw new InvalidDataException("Fixture metadata differs");
    tensors.Add(f.Name,array);
}
var referenced=new HashSet<string>();
foreach(var c in manifest.GetProperty("cases").EnumerateArray())
{
    foreach(string key in new[]{"input","output","raw","windowed","power"})if(!referenced.Add(c.GetProperty(key).GetString()??""))throw new InvalidDataException("Reused case fixture");
    var input=tensors[c.GetProperty("input").GetString()??""];int frames=1+(input.Values.Length-400)/160;
    if(input.Shape.Length!=1 || !tensors[c.GetProperty("output").GetString()??""].Shape.SequenceEqual(new[]{1,frames,80}))throw new InvalidDataException("Case contract differs");
}
var reports=new List<object>();long values=0;double maximum=0;bool passed=true;
foreach(var c in manifest.GetProperty("cases").EnumerateArray())
{
    string name=c.GetProperty("name").GetString()??throw new InvalidDataException();var input=tensors[c.GetProperty("input").GetString()??""];var expected=tensors[c.GetProperty("output").GetString()??""];
    string before=Sha(MemoryMarshal.AsBytes(input.Values.AsSpan()).ToArray());var result=WeSpeakerAudio.LogMelFilterbank(input.Values,16000,CancellationToken.None);
    if(!result.Dimensions.SequenceEqual(expected.Shape))throw new InvalidDataException("Shape");
    var actual=result.ToArray();double error=0;int bad=0,worst=0;
    for(int i=0;i<actual.Length;i++){if(!float.IsFinite(actual[i]))throw new InvalidDataException("Nonfinite");double e=Math.Abs((double)actual[i]-expected.Values[i])/Math.Max(1,Math.Abs((double)expected.Values[i]));if(e>error){error=e;worst=i;}if(e>1e-4)bad++;}
    byte[] bytes=MemoryMarshal.AsBytes(actual.AsSpan()).ToArray();string file=name+".f32";File.WriteAllBytes(Path.Combine(destination+".arrays",file),bytes);
    if(before!=Sha(MemoryMarshal.AsBytes(input.Values.AsSpan()).ToArray())||!actual.SequenceEqual(WeSpeakerAudio.LogMelFilterbank(input.Values,16000,CancellationToken.None).ToArray()))throw new InvalidDataException("Ownership/repeat");
    reports.Add(new{name,file,shape=expected.Shape,sha256=Sha(bytes),error,bad,worst_index=worst,actual=actual[worst],expected=expected.Values[worst]});values+=actual.Length;maximum=Math.Max(maximum,error);passed&=bad==0;
    Console.WriteLine($"{name}: {error:R}, bad={bad}");
}
using var process=Process.GetCurrentProcess();foreach(ProcessModule m in process.Modules)if(m.ModuleName.Contains("onnxruntime",StringComparison.OrdinalIgnoreCase))throw new InvalidDataException("Native runtime loaded");
File.WriteAllText(destination,JsonSerializer.Serialize(new{passed,values,maximum,reports,manifest_sha256=Sha(File.ReadAllBytes(Path.Combine(root,"manifest.json"))),data_sha256=Sha(File.ReadAllBytes(typeof(WeSpeakerAudio).Assembly.Location)),core_sha256=Sha(File.ReadAllBytes(typeof(Tensor<float>).Assembly.Location)),runtime=Environment.Version.ToString(),peak=process.PeakWorkingSet64,
    settings=Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>().Where(p=>p.Key.ToString()?.StartsWith("LOKAD_ONNX_",StringComparison.Ordinal)==true||p.Key.ToString()?.StartsWith("DOTNET_",StringComparison.Ordinal)==true).ToDictionary(p=>p.Key.ToString()??"",p=>p.Value?.ToString())},new JsonSerializerOptions{WriteIndented=true}));
return passed?0:1;
