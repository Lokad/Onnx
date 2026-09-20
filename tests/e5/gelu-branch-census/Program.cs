using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

static void Require(bool ok, string why) { if (!ok) throw new InvalidDataException(why); }
static string Hash(string file) { using var s=File.OpenRead(file); return Convert.ToHexStringLower(SHA256.HashData(s)); }
static void Save(string file, ReadOnlySpan<float> data)
{
    using var s=new FileStream(file,FileMode.CreateNew);s.Write(MemoryMarshal.AsBytes(data));
}
static void Json(string file, object value)
{
    using var s=new FileStream(file,FileMode.CreateNew);
    JsonSerializer.Serialize(s,value,new JsonSerializerOptions{WriteIndented=true});
}
string model=Path.GetFullPath(args[0]),fixtures=Path.GetFullPath(args[1]),destination=Path.GetFullPath(args[2]);
Require(!Directory.Exists(destination),"Destination exists");
if (!OperatingSystem.IsWindows()) throw new PlatformNotSupportedException("This capture profile requires Windows");
Require(Environment.ProcessorCount==1 && Process.GetCurrentProcess().ProcessorAffinity.ToInt64()==4,"CPU2 required before CLR startup");
Require(Vector<float>.Count==8,"Eight-lane production route required");
string core=Hash(typeof(ComputationalGraph).Assembly.Location);
Require(core=="187de61ad8f034b9b7ad2fb3490358443fa84334204720e81bc3546a31f3c8d4","Wrong core");
string modelHash=Hash(model);
Require(modelHash=="ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665","Wrong model");
var settings=Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k=>k.StartsWith("LOKAD_",StringComparison.Ordinal)||k.StartsWith("DOTNET_",StringComparison.Ordinal)||k.StartsWith("COMPlus_",StringComparison.Ordinal)).ToDictionary(k=>k,Environment.GetEnvironmentVariable);
Require(settings.Count==0,"Runtime overrides present");
Directory.CreateDirectory(destination);
var graph=OnnxImport.Load(model) ?? throw new InvalidDataException("Import failed");
var opts=ExecutionOptions.Default with { Tensor=ExecutionOptions.Default.Tensor with { DisableBufferPool=true } };
var all=new List<object>();
foreach(string name in new[]{"e5-8tok","e5-30tok","e5-30pad128","e5-128tok","e5-512tok"})
{
    using var doc=JsonDocument.Parse(File.ReadAllBytes(Path.Combine(fixtures,name+".json")));var fixture=doc.RootElement;
    Require(fixture.GetProperty("name").GetString()==name && fixture.GetProperty("model_sha256").GetString()==modelHash,"Fixture identity");
    var values=fixture.GetProperty("inputs").EnumerateObject().ToDictionary(p=>p.Name,p=>p.Value.EnumerateArray().Select(x=>x.GetInt64()).ToArray());
    var inputs=values.ToDictionary(p=>p.Key,p=>(ITensor)new DenseTensor<long>(p.Value.ToArray(),new[]{1,p.Value.Length}));
    graph.Reset();Require(graph.Execute(inputs,true,ExecutionProvider.CPU,opts),graph.LastErrorMessage ?? "Execute failed");
    var output=(Tensor<float>)graph.Outputs["last_hidden_state"];
    var actual=output.ToArray();
    string referencePath=Path.Combine(fixtures,fixture.GetProperty("reference_file").GetString()!);
    Require(Hash(referencePath)==fixture.GetProperty("reference_sha256").GetString(),"Reference hash");
    var reference=MemoryMarshal.Cast<byte,float>(File.ReadAllBytes(referencePath)).ToArray();
    Require(output.Dimensions.SequenceEqual(fixture.GetProperty("shape").EnumerateArray().Select(x=>x.GetInt32()).ToArray()) && actual.Length==reference.Length,"Output geometry");
    double maxError=0;
    for(int i=0;i<actual.Length;i++)
    {
        Require(float.IsFinite(actual[i]) && float.IsFinite(reference[i]),"Nonfinite final output");
        maxError=Math.Max(maxError,Math.Abs((double)actual[i]-reference[i])/Math.Max(1,Math.Abs((double)reference[i])));
    }
    Require(maxError<=1e-4,"Native output gate");
    string folder=Path.Combine(destination,name);Directory.CreateDirectory(folder);
    Save(Path.Combine(folder,"output.f32"),actual);
    var rows=new List<object>();
    var nodes=graph.Nodes.Where(n=>n.Op==OpType.BiasGelu).ToArray();Require(nodes.Length==12,"Expected twelve BiasGelu nodes");
    for(int index=0;index<nodes.Length;index++)
    {
        var node=nodes[index];
        var x=(Tensor<float>)graph.GetInputTensor(node.Inputs[0])!;
        var bias=(Tensor<float>)graph.GetInputTensor(node.Inputs[1])!;
        var y=(Tensor<float>)graph.GetInputTensor(node.Outputs[0])!;
        var xs=x.ToArray();var bs=bias.ToArray();var ys=y.ToArray();
        Require(bs.Length==1536 && xs.Length==values["input_ids"].Length*1536 && ys.Length==xs.Length,"GELU geometry");
        var replay=CPUExecutionProvider.BiasGelu(x,bias,ExecutionOptions.Default,null);
        Require(replay.Status==OpStatus.Success,"GELU replay failed");
        var replayValues=((Tensor<float>)replay.Outputs![0]).ToArray();
        Require(MemoryMarshal.AsBytes(ys.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(replayValues.AsSpan())),"Captured GELU output differs from actual product replay");
        long small=0,allSmall=0,allLarge=0;int biasIndex=0;
        for(int i=0;i<xs.Length;i+=8)
        {
            var scaled=(new Vector<float>(xs,i)+new Vector<float>(bs,biasIndex))*new Vector<float>(.7071067811865476f);
            int count=0;
            for(int lane=0;lane<8;lane++){float value=scaled[lane];Require(float.IsFinite(value),"Nonfinite GELU argument");if(MathF.Abs(value)<=.921875f)count++;}
            small+=count;if(count==8)allSmall++;if(count==0)allLarge++;
            biasIndex+=8;if(biasIndex==bs.Length)biasIndex=0;
        }
        string prefix=index.ToString("D2");
        Save(Path.Combine(folder,prefix+"-x.f32"),xs);Save(Path.Combine(folder,prefix+"-bias.f32"),bs);Save(Path.Combine(folder,prefix+"-y.f32"),ys);
        rows.Add(new{index,id=node.ID,name=node.Name,inputs=node.Inputs,outputs=node.Outputs,shape=x.Dimensions.ToArray(),bias_shape=bias.Dimensions.ToArray(),values=xs.Length,vectors=xs.Length/8,small_lanes=small,all_small=allSmall,all_large=allLarge,mixed=xs.Length/8-allSmall-allLarge,product_replay_bitwise=true});
    }
    foreach(var pair in inputs)Require(((Tensor<long>)pair.Value).ToArray().SequenceEqual(values[pair.Key]),"Input mutated");
    var result=new{name,fixture_sha256=Hash(Path.Combine(fixtures,name+".json")),input_sha256=fixture.GetProperty("input_sha256").GetString(),max_native_error=maxError,output_shape=output.Dimensions.ToArray(),inputs_unchanged=true,nodes=rows};
    Json(Path.Combine(folder,"capture.json"),result);all.Add(result);Console.WriteLine("Captured "+name+": 12 GELU nodes, native error "+maxError);
}
Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m=>Path.GetFileName(m.FileName).StartsWith("onnxruntime",StringComparison.OrdinalIgnoreCase)),"Native ORT loaded");
var files=Directory.GetFiles(destination,"*",SearchOption.AllDirectories).ToDictionary(f=>Path.GetRelativePath(destination,f).Replace('\\','/'),f=>new{bytes=new FileInfo(f).Length,sha256=Hash(f)});
Json(Path.Combine(destination,"capture.json"),new{schema=1,diagnostic="uniform-gelu-branch-census",core_sha256=core,model_sha256=modelHash,probe_sha256=Hash(typeof(Program).Assembly.Location),runtime=RuntimeInformation.FrameworkDescription,affinity=4,vector_width=Vector<float>.Count,settings,cases=all,files});
