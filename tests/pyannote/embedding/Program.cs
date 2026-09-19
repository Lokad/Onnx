using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

if (args.Length != 3) throw new ArgumentException("EmbeddingReplay <encoder.onnx> <manifest.json> <new-result.json>");
string modelPath=Path.GetFullPath(args[0]), manifestPath=Path.GetFullPath(args[1]), destination=Path.GetFullPath(args[2]);
string root=Path.GetDirectoryName(manifestPath) ?? throw new InvalidDataException();
if (File.Exists(destination) || Directory.Exists(destination+".tensors")) throw new IOException("Output exists");
var reports=new List<object>(); var held=new List<(Tensor<float> Tensor,string Sha)>();var errors=new List<string>();
var watch=Stopwatch.StartNew();bool passed=true;long values=0;double maximum=0;int rejections=0;
string Sha(string file) { using var stream=File.OpenRead(file);return Convert.ToHexStringLower(SHA256.HashData(stream)); }
string SourceSha(byte[] bytes)=>Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(Encoding.UTF8.GetString(bytes).Replace("\r\n","\n",StringComparison.Ordinal))));
byte[] Bytes(Tensor<float> value)=>MemoryMarshal.AsBytes(value.ToArray().AsSpan()).ToArray();
string Bits(Tensor<float> value)=>Convert.ToHexStringLower(SHA256.HashData(Bytes(value)));
void Require(bool condition,string message) { if (!condition) throw new InvalidDataException(message); }
void Hold(Tensor<float> value)=>held.Add((value,Bits(value)));
void CheckHeld() { foreach(var item in held) Require(Bits(item.Tensor)==item.Sha,"Held input/output changed"); }
void NoNative() { using var process=Process.GetCurrentProcess();foreach(ProcessModule m in process.Modules) Require(!m.ModuleName.Contains("onnxruntime",StringComparison.OrdinalIgnoreCase),"Native runtime loaded"); }
try
{
    NoNative();Require(Sha(modelPath)=="9903474d6230e5e858dc6b6382a0e3f6e402ea9b4210e1e2f2bee60a33830e7a","Model digest differs");
    using var doc=JsonDocument.Parse(File.ReadAllText(manifestPath));var manifest=doc.RootElement;
    Require(manifest.GetProperty("scope").GetString()=="pyannote embedding backbone, not full embedding or diarization"
        && manifest.GetProperty("scaled_absolute_tolerance").GetDouble()==1e-4,"Scope or gate differs");
    foreach(var version in new[]{("numpy","2.2.4"),("onnxruntime","1.29.0"),("torch","2.11.0+cpu"),("torchaudio","2.11.0+cpu")})
        Require(manifest.GetProperty(version.Item1).GetString()==version.Item2,"Native versions differ");
    using var pins=Assembly.GetExecutingAssembly().GetManifestResourceStream("embedding-assets.json") ?? throw new InvalidDataException();
    using var pinBytes=new MemoryStream();pins.CopyTo(pinBytes);
    Require(JsonNode.DeepEquals(JsonNode.Parse(pinBytes.ToArray()),JsonNode.Parse(manifest.GetProperty("assets").GetRawText())),"Asset pins differ");
    Require(SourceSha(pinBytes.ToArray())==manifest.GetProperty("assets_lf_sha256").GetString(),"Asset source differs");
    using var generator=Assembly.GetExecutingAssembly().GetManifestResourceStream("embedding-generator.py") ?? throw new InvalidDataException();
    using var generatorBytes=new MemoryStream();generator.CopyTo(generatorBytes);
    Require(SourceSha(generatorBytes.ToArray())==manifest.GetProperty("generator_lf_sha256").GetString(),"Generator source differs");
    var native=manifest.GetProperty("native_settings");
    Require(native.GetProperty("provider").GetString()=="CPUExecutionProvider" && native.GetProperty("threads").GetInt32()==1
        && native.GetProperty("execution").GetString()=="sequential" && native.GetProperty("optimization").GetString()=="all"
        && !native.GetProperty("spinning").GetBoolean(),"Native settings differ");
    Require(JsonNode.DeepEquals(JsonNode.Parse(manifest.GetProperty("frontend_settings").GetRawText()),JsonNode.Parse(
        "{\"num_mel_bins\":80,\"frame_length\":25.0,\"frame_shift\":10.0,\"round_to_power_of_two\":true,\"snip_edges\":true,\"dither\":0.0,\"sample_frequency\":16000,\"window_type\":\"hamming\",\"use_energy\":false}")),"Frontend settings differ");
    var tensors=new Dictionary<string,Tensor<float>>();
    foreach(var entry in manifest.GetProperty("files").EnumerateObject())
    {
        string path=Path.GetFullPath(Path.Combine(root,entry.Name));Require(Path.GetDirectoryName(path)==root,"Fixture path differs");
        Require(Sha(path)==entry.Value.GetProperty("sha256").GetString() && new FileInfo(path).Length==entry.Value.GetProperty("bytes").GetInt64(),"Fixture digest differs");
        Require(entry.Value.GetProperty("dtype").GetString()=="float32","Fixture dtype differs");
        var tensor=NpySupport.ReadTensor(path) as Tensor<float> ?? throw new InvalidDataException();
        Require(tensor.Dimensions.SequenceEqual(entry.Value.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt32()).ToArray()),"Fixture shape differs");
        Require(tensor.ToArray().All(float.IsFinite),"Nonfinite native fixture");tensors.Add(entry.Name,tensor);
    }
    var cases=manifest.GetProperty("cases").EnumerateArray().ToArray();
    string[] names=["synthetic-b1-f200","synthetic-b1-f201","synthetic-b1-f400","synthetic-b1-f800","synthetic-b2-f200","english-16k","french-44k-stereo","jfk-48k-stereo"];
    Require(cases.Select(c=>c.GetProperty("name").GetString()).SequenceEqual(names),"Coverage differs");
    Require(tensors.Count==16 && manifest.GetProperty("rejections").EnumerateArray().Select(r=>r.GetProperty("name").GetString()).SequenceEqual(new[]{"missing","rank","width"}),"Fixture/rejection coverage differs");
    int[] lengths=[200,201,400,800,200,584,622,1098];var referenced=new HashSet<string>();
    for(int i=0;i<cases.Length;i++)
    {
        string input=cases[i].GetProperty("input").GetString() ?? "",output=cases[i].GetProperty("output").GetString() ?? "";
        Require(referenced.Add(input)&&referenced.Add(output),"Reused fixture binding");
        Require(tensors[input].Dimensions.SequenceEqual(new[]{i==4?2:1,lengths[i],80})
            && tensors[output].Dimensions.SequenceEqual(new[]{i==4?2:1,2560,(lengths[i]+7)/8}),"Case shape differs");
    }
    Directory.CreateDirectory(destination+".tensors");
    var graph=OnnxImport.Load(modelPath,64L*1024*1024) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
    var context=graph.CreateExecution(ExecutionOptions.Memory);
    var ownHashes=new Dictionary<string,string>();
    for(int repeat=0;repeat<2;repeat++) foreach(var c in cases)
    {
        string name=c.GetProperty("name").GetString() ?? throw new InvalidDataException();
        var input=tensors[c.GetProperty("input").GetString() ?? ""];Hold(input);
        var expected=tensors[c.GetProperty("output").GetString() ?? ""];
        var feeds=new Dictionary<string,ITensor>{{"fbank_features",input}};
        graph.Reset();context.Reset();CheckHeld();
        bool success=repeat==0 ? graph.Execute(feeds,true,ExecutionProvider.CPU,ExecutionOptions.Memory) : context.Execute(feeds,true,ExecutionProvider.CPU,ExecutionOptions.Memory);
        Require(success,(repeat==0?graph.LastErrorMessage:context.LastErrorMessage) ?? "Execution failed");
        var outputs=repeat==0?graph.Outputs:context.Outputs;
        var output=outputs["/resnet/pool/Reshape_output_0"] as Tensor<float> ?? throw new InvalidDataException();
        Require(output.Dimensions.SequenceEqual(expected.Dimensions),"Output shape differs");
        float[] a=output.ToArray(),b=expected.ToArray();double error=0;int worst=0,bad=0;
        for(int i=0;i<a.Length;i++)
        {
            Require(float.IsFinite(a[i]),"Nonfinite managed output");
            double d=Math.Abs((double)a[i]-b[i])/Math.Max(1,Math.Abs((double)b[i]));
            if(d>error) { error=d;worst=i; } if(d>1e-4)bad++;
        }
        string bits=Bits(output);if(repeat==0)ownHashes.Add(name,bits);else Require(ownHashes[name]==bits,"Repeat differs");
        string file=name+"-"+repeat+".f32";File.WriteAllBytes(Path.Combine(destination+".tensors",file),Bytes(output));
        reports.Add(new{name,repeat,file,sha256=bits,shape=output.Dimensions.ToArray(),max_error=error,worst_index=worst,bad,passed=bad==0});
        Hold(output);values+=a.Length;maximum=Math.Max(maximum,error);passed&=bad==0;CheckHeld();
        Console.WriteLine(name+" repeat "+repeat+" max "+error.ToString("R"));
    }
    foreach(var feed in new Dictionary<string,ITensor>[] {new(),new(){{"fbank_features",new DenseTensor<float>(new[]{200,80})}},new(){{"fbank_features",new DenseTensor<float>(new[]{1,200,79})}}})
    {
        context.Reset();Require(!context.Execute(feed,true,ExecutionProvider.CPU,ExecutionOptions.Memory),"Invalid input accepted");
        Require(!string.IsNullOrWhiteSpace(context.LastErrorMessage),"Failure lacks error");rejections++;CheckHeld();
    }
    context.Reset();var first=tensors[cases[0].GetProperty("input").GetString() ?? ""];
    Require(context.Execute(new Dictionary<string,ITensor>{{"fbank_features",first}},true,ExecutionProvider.CPU,ExecutionOptions.Memory),"Recovery failed");
    Require(Bits((Tensor<float>)(context.Outputs["/resnet/pool/Reshape_output_0"] ?? throw new InvalidDataException()))==ownHashes[cases[0].GetProperty("name").GetString() ?? ""],"Recovery output differs");
    graph.Reset();context.Reset();CheckHeld();NoNative();
}
catch(Exception ex){passed=false;errors.Add(ex.ToString());Console.Error.WriteLine(ex);}
var report=new{scope="Full-array pyannote embedding backbone; no final embedding/diarization claim",passed,values,max_error=maximum,rejections,
    manifest_sha256=Sha(manifestPath),model_sha256=Sha(modelPath),core_sha256=Sha(typeof(ComputationalGraph).Assembly.Location),runner_sha256=Sha(Assembly.GetExecutingAssembly().Location),
    runtime=RuntimeInformation.FrameworkDescription,
    settings=Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>()
        .Where(p=>p.Key is string k&&(k.StartsWith("DOTNET_",StringComparison.OrdinalIgnoreCase)||k.StartsWith("COMPlus_",StringComparison.OrdinalIgnoreCase)||k.StartsWith("LOKAD_ONNX_",StringComparison.OrdinalIgnoreCase)))
        .ToDictionary(p=>(string)p.Key,p=>p.Value?.ToString()),
    peak_working_set=Process.GetCurrentProcess().PeakWorkingSet64,seconds=watch.Elapsed.TotalSeconds,held=held.Count,reports,errors};
using(var stream=new FileStream(destination,FileMode.CreateNew,FileAccess.Write))JsonSerializer.Serialize(stream,report,new JsonSerializerOptions{WriteIndented=true});
return passed?0:1;
