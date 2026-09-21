using System.Diagnostics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

Require(args.Length == 3, "root manifest role");
string root = Path.GetFullPath(args[0]), manifest = Path.GetFullPath(args[1]), role = args[2];
using var document = JsonDocument.Parse(File.ReadAllBytes(manifest)); var spec = document.RootElement;
Require(spec.GetProperty("protocol").GetString() == "e5-profiler-allocation-v1", "Protocol");
Require(role is "baseline" or "candidate", "Role");
Require(OperatingSystem.IsWindows() && Environment.Version.ToString() == "10.0.12"
    && Environment.ProcessorCount == 1 && Process.GetCurrentProcess().ProcessorAffinity.ToInt64() == 4, "Runtime/CPU");
Require(!Environment.GetEnvironmentVariables().Keys.Cast<string>().Any(k => k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase)
    || k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)), "Runtime overrides");
string core = Sha(typeof(ComputationalGraph).Assembly.Location);
Require(core == spec.GetProperty("cores").GetProperty(role).GetProperty("sha256").GetString(), "Core");
string folder = Path.Combine(root, spec.GetProperty("base").GetString()!, "outputs", role);
Require(!Directory.Exists(folder), "Output exists"); Directory.CreateDirectory(folder);
NoNative();
Write(Path.Combine(folder,"il.json"), Inspect());
var rows = new List<object>(); var groups = new List<object>();
foreach (string name in spec.GetProperty("cases").EnumerateArray().Select(v => v.GetString()!))
{
    string fixtures = Path.Combine(root,spec.GetProperty("fixtures").GetString()!);
    using var fixtureDoc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(fixtures,name+".json")));
    var fixture = fixtureDoc.RootElement;
    string model = Path.Combine(root,"models/multilingual-e5-small/model.onnx");
    Require(Sha(model) == fixture.GetProperty("model_sha256").GetString(), "Model identity");
    string reference = Path.Combine(fixtures,fixture.GetProperty("reference_file").GetString()!);
    Require(Sha(reference) == fixture.GetProperty("reference_sha256").GetString(), "Reference identity");
    float[] wanted = MemoryMarshal.Cast<byte,float>(File.ReadAllBytes(reference)).ToArray();
    int[] shape = fixture.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
    var feed = fixture.GetProperty("inputs").EnumerateObject().ToDictionary(p => p.Name,p =>
        (ITensor)new DenseTensor<long>(p.Value.EnumerateArray().Select(v => v.GetInt64()).ToArray(),new[]{1,shape[1]}));
    var before = feed.ToDictionary(p => p.Key,p => TensorHash(p.Value));
    var graph = OnnxImport.Load(model) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
    graph.Prepare();
    foreach (bool memory in new[]{false,true})
    foreach (string mode in new[]{"disabled","detailed","wall"})
    {
        string key = name+"-"+(memory?"memory":"default")+"-"+mode;
        string groupFolder = Path.Combine(folder,key); Directory.CreateDirectory(groupFolder);
        var options = memory?ExecutionOptions.Memory:ExecutionOptions.Default;
        var context = graph.CreateExecution(options);
        string? expectedBits = null, profileDescription = null; ITensor? held = null; string? heldBits = null;
        var nodes = context.Nodes.Select(n => new { id=n.ID, op=n.Op.ToString() }).ToArray();
        int warmup = spec.GetProperty("warmup").GetInt32(), observed = spec.GetProperty("observed").GetInt32();
        for (int call = 0; call < warmup+observed; call++)
        {
            using var outer = mode == "wall" ? Profiler.BeginWallExecution() : Profiler.BeginExecution(mode == "detailed");
            long allocated = GC.GetAllocatedBytesForCurrentThread();
            bool ok = context.Execute(feed,false,ExecutionProvider.CPU,options);
            long publicAllocated = GC.GetAllocatedBytesForCurrentThread()-allocated;
            Require(ok,context.LastErrorMessage ?? "Execution failed");
            long runAllocated = context.LastAllocatedBytes;
            Require(context.Outputs.Count == 1 && context.Outputs.ContainsKey("last_hidden_state"),"Output names");
            var tensor = (Tensor<float>)context.Outputs["last_hidden_state"];
            Require(tensor.Dimensions.SequenceEqual(shape),"Output shape");
            float[] values = tensor.ToArray(); double maximum = 0;
            for (int i=0;i<values.Length;i++)
            {
                Require(float.IsFinite(values[i]) && float.IsFinite(wanted[i]),"Nonfinite");
                maximum = Math.Max(maximum,Math.Abs((double)values[i]-wanted[i])/Math.Max(1.0,Math.Abs((double)wanted[i])));
            }
            Require(maximum<=1e-4,"Native gate");
            string bits = TensorHash(tensor); expectedBits ??= bits; Require(bits==expectedBits,"Repeat bits");
            Require(feed.All(p => TensorHash(p.Value)==before[p.Key]),"Input mutation");
            if (held is not null) Require(TensorHash(held)==heldBits,"Held output mutation");
            if (call==0) { held=tensor;heldBits=bits; }
            object description;
            if (mode=="detailed")
            {
                var profile=context.LastProfile!.Reverse().ToArray();
                Require(profile.Length==nodes.Length,"Detailed count");
                for(int i=0;i<profile.Length;i++)
                {
                    Require(profile[i].NodeId==nodes[i].id && profile[i].Op.ToString()==nodes[i].op,"Detailed node");
                    Require(profile[i].Detail is not null && profile[i].OpsProfile.Count>0,"Detailed payload");
                    Require(profile[i].OpsProfile.All(p=>p.Time>=TimeSpan.Zero),"Stage time");
                }
                Require(context.LastWallProfile!.Count==0,"Unexpected wall profile");
                description=profile.Select(p=>new {p.NodeId,op=p.Op.ToString(),p.Detail,stages=p.OpsProfile.Select(o=>o.Stage.ToString()).ToArray()}).ToArray();
            }
            else if(mode=="wall")
            {
                var profile=context.LastWallProfile!.ToArray();
                Require(profile.Length==nodes.Length && context.LastProfile!.Count==0,"Wall count");
                for(int i=0;i<profile.Length;i++)
                {
                    Require(profile[i].NodeId==nodes[i].id && profile[i].Op.ToString()==nodes[i].op,"Wall node");
                    Require(profile[i].StartTicks>0 && profile[i].EndTicks>=profile[i].StartTicks,"Wall range");
                    if(i>0)Require(profile[i].StartTicks>=profile[i-1].EndTicks,"Wall ordering");
                }
                description=profile.Select(p=>new {p.NodeId,op=p.Op.ToString()}).ToArray();
            }
            else
            {
                Require(context.LastProfile!.Count==0 && context.LastWallProfile!.Count==0,"Disabled profile");
                description=Array.Empty<object>();
            }
            string detail=JsonSerializer.Serialize(description);profileDescription??=detail;
            Require(detail==profileDescription,"Profile contents changed");
            if(call==0 || call==warmup+observed-1)
            {
                using var stream=new FileStream(Path.Combine(groupFolder,call+".f32"),FileMode.CreateNew);
                stream.Write(MemoryMarshal.AsBytes(values.AsSpan()));
            }
            rows.Add(new { key,call,warmup=call<warmup,run_allocated=runAllocated,public_allocated=publicAllocated,
                nodes=nodes.Length,sha256=bits,maximum_native_error=maximum,profile_sha256=Convert.ToHexStringLower(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(detail))) });
            context.Reset();Require(TensorHash(tensor)==bits && TensorHash(held!)==heldBits,"Reset mutation");
        }
        File.WriteAllText(Path.Combine(groupFolder,"profile.json"),profileDescription);
        groups.Add(new { key,nodes=nodes.Length,shape,output_sha256=expectedBits });
        Console.WriteLine(key+" complete");
    }
}
NoNative();
Write(Path.Combine(folder,"result.json"),new { complete=true,role,core_sha256=core,manifest_sha256=Sha(manifest),
    runtime=Environment.Version.ToString(),affinity=new[]{2},processor_count=Environment.ProcessorCount,
    inputs_unchanged=true,held_outputs_unchanged=true,native_loaded=false,groups,rows });

static object Inspect()
{
    var method=typeof(ComputationalGraph).GetMethod("RunCoreInner",BindingFlags.Instance|BindingFlags.NonPublic)!;
    var body=method.GetMethodBody()!;byte[] code=body.GetILAsByteArray()!;
    var codes=typeof(OpCodes).GetFields(BindingFlags.Public|BindingFlags.Static).Where(f=>f.FieldType==typeof(OpCode))
        .Select(f=>(OpCode)f.GetValue(null)!).ToDictionary(v=>unchecked((ushort)v.Value));
    var instructions=new List<object>();int offset=0;
    while(offset<code.Length)
    {
        int start=offset;ushort value=code[offset++];if(value==0xfe)value=(ushort)(0xfe00|code[offset++]);
        var op=codes[value];object operand="";int size;
        switch(op.OperandType)
        {
            case OperandType.InlineNone:break;
            case OperandType.InlineMethod:case OperandType.InlineField:case OperandType.InlineType:case OperandType.InlineTok:
                var member=method.Module.ResolveMember(BitConverter.ToInt32(code,offset))!;
                operand=member.DeclaringType+"::"+member;offset+=4;break;
            case OperandType.InlineString:operand=method.Module.ResolveString(BitConverter.ToInt32(code,offset));offset+=4;break;
            case OperandType.ShortInlineBrTarget:operand=offset+1+(sbyte)code[offset];offset++;break;
            case OperandType.InlineBrTarget:operand=offset+4+BitConverter.ToInt32(code,offset);offset+=4;break;
            case OperandType.InlineSwitch:
                int count=BitConverter.ToInt32(code,offset);offset+=4;int next=offset+4*count;
                operand=Enumerable.Range(0,count).Select(i=>next+BitConverter.ToInt32(code,offset+4*i)).ToArray();offset=next;break;
            default:
                size=op.OperandType is OperandType.ShortInlineI or OperandType.ShortInlineVar?1:
                    op.OperandType==OperandType.InlineVar?2:op.OperandType is OperandType.InlineI8 or OperandType.InlineR?8:4;
                operand=Convert.ToHexString(code.AsSpan(offset,size));offset+=size;break;
        }
        instructions.Add(new {offset=start,opcode=op.Name,operand});
    }
    return new { method=method.ToString(),body.MaxStackSize,locals=body.LocalVariables.Select(v=>v.LocalType.ToString()).ToArray(),instructions };
}
static void Write(string path,object value){using var f=new FileStream(path,FileMode.CreateNew);JsonSerializer.Serialize(f,value,new JsonSerializerOptions{WriteIndented=true});}
static string Sha(string path){using var f=File.OpenRead(path);return Convert.ToHexStringLower(SHA256.HashData(f));}
static string TensorHash(ITensor value)=>value switch{
    Tensor<float> f=>Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(f.ToArray().AsSpan()))),
    Tensor<long> l=>Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(l.ToArray().AsSpan()))),
    _=>throw new InvalidDataException("Tensor type")};
static void NoNative()=>Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m=>m.ModuleName.Contains("onnxruntime",StringComparison.OrdinalIgnoreCase)),"Native loaded");
static void Require(bool ok,string why){if(!ok)throw new InvalidDataException(why);}
