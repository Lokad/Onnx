using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;

static class Screen
{
    const string ComponentHash = "ffedd38708b086c2620ac27f8ab540880e59d7d66b8a65f050447a80b2f88377";
    delegate float[] Prepare(ReadOnlySpan<float> weights,int c,int m,int lanes);
    delegate bool Direct(ReadOnlySpan<float> input,ReadOnlySpan<float> weights,ReadOnlySpan<float> bias,
        ReadOnlySpan<float> residual,Span<float> destination,Span<float> packedInput,Span<float> packedOutput,
        int c,int m,int h,int w,int stride,int lanes,bool relu);
    delegate bool Winograd(ReadOnlySpan<float> input,ReadOnlySpan<float> weights,ReadOnlySpan<float> bias,
        ReadOnlySpan<float> residual,Span<float> destination,Span<float> transformed,Span<float> products,Span<float> blocked,
        int c,int m,int h,int w,int lanes,bool relu);
    delegate bool Plan(int c,int m,int h,int w,out int input,out int products,out int output);
    sealed record Case(string Fixture,int Index,int Form,int C,int M,int H,int W,bool Relu,
        float[] Input,float[] Weights,float[] Bias,float[] Residual,string CurrentHash,string CandidateHash)
    {
        internal long Work => (long)M*C*9*H*W;
        internal int Iterations => (int)Math.Max(1,((1L<<31)+Work-1)/Work);
    }
    static void Require(bool value,string reason) { if (!value) throw new InvalidOperationException(reason); }
    static string Hash(float[] values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
    static string FileHash(string file) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(file)));
    static T Bind<T>(Type type,string name) where T:Delegate => type.GetMethod(name,BindingFlags.Static|BindingFlags.NonPublic)!.CreateDelegate<T>();

    static float[] CompleteCall(Case call,float[] prepared,bool candidate,Direct direct,Winograd winograd,Plan plan,
        out long requested,out long rented)
    {
        int outputCount=checked(call.M*call.H*call.W);
        int ni,np,no;
        if (candidate)
            Require(plan(call.C,call.M,call.H,call.W,out ni,out np,out no),"candidate scratch budget");
        else { ni=checked(call.C*(call.H+2)*(call.W+2));np=outputCount;no=0; }
        requested=checked(4L*(ni+np+no));Require(requested<=64L*1024*1024,"scratch budget");
        var output=new float[outputCount];float[]? a=null,b=null,d=null;rented=0;
        try
        {
            a=ArrayPool<float>.Shared.Rent(ni);b=ArrayPool<float>.Shared.Rent(np);
            if (candidate) d=ArrayPool<float>.Shared.Rent(no);
            rented=4L*(a.Length+b.Length+(d?.Length??0));
            bool ok=candidate
                ? winograd(call.Input,prepared,call.Bias,call.Residual,output,a,b,d!,call.C,call.M,call.H,call.W,16,call.Relu)
                : direct(call.Input,prepared,call.Bias,call.Residual,output,a,b,call.C,call.M,call.H,call.W,1,16,call.Relu);
            Require(ok,"qualified call refused");return output;
        }
        finally
        {
            if (d is not null) ArrayPool<float>.Shared.Return(d);
            if (b is not null) ArrayPool<float>.Shared.Return(b);
            if (a is not null) ArrayPool<float>.Shared.Return(a);
        }
    }

    static List<Case> Load(string folder,string reference,Dictionary<string,float[]> tensors)
    {
        using var refs=JsonDocument.Parse(File.ReadAllText(reference));
        var expected=refs.RootElement.GetProperty("rows").EnumerateArray().ToDictionary(
            x=>(x.GetProperty("fixture").GetString()!,x.GetProperty("index").GetInt32()),
            x=>(x.GetProperty("selectedOutput").GetString()!,x.GetProperty("output").GetString()!));
        using var doc=JsonDocument.Parse(File.ReadAllText(Path.Combine(folder,"result.json")));
        float[] Tensor(JsonElement e)
        {
            if (e.ValueKind==JsonValueKind.Null) return Array.Empty<float>();
            string name=e.GetProperty("path").GetString()!;
            Require(Path.GetFileName(name)==name,"fixture path");
            if (!tensors.TryGetValue(name,out var data))
            {
                byte[] bytes=File.ReadAllBytes(Path.Combine(folder,name));
                Require(bytes.Length==e.GetProperty("bytes").GetInt32()&&Convert.ToHexStringLower(SHA256.HashData(bytes))==e.GetProperty("sha256").GetString(),"fixture digest");
                data=MemoryMarshal.Cast<byte,float>(bytes).ToArray();tensors.Add(name,data);
            }
            return data;
        }
        var calls=new List<Case>();
        foreach (var call in doc.RootElement.GetProperty("calls").EnumerateArray())
        {
            if (!call.GetProperty("eligible").GetBoolean()||!call.GetProperty("attributes").GetProperty("strides").EnumerateArray().Select(x=>x.GetInt32()).SequenceEqual(new[]{1,1})) continue;
            int[] shape=call.GetProperty("input").GetProperty("shape").EnumerateArray().Select(x=>x.GetInt32()).ToArray();
            int m=call.GetProperty("weights").GetProperty("shape")[0].GetInt32();
            string fixture=call.GetProperty("case").GetString()!;int index=call.GetProperty("index").GetInt32();
            var hashes=expected[(fixture,index)];
            calls.Add(new(fixture,index,call.GetProperty("form").GetInt32(),shape[1],m,shape[2],shape[3],call.GetProperty("relu").GetBoolean(),
                Tensor(call.GetProperty("input")),Tensor(call.GetProperty("weights")),Tensor(call.GetProperty("bias")),Tensor(call.GetProperty("residual")),hashes.Item1,hashes.Item2));
        }
        Require(calls.Count==87&&calls.All(c=>c.Iterations==3),"fixed geometry census");
        Require(calls.Select(c=>(c.Fixture,c.Index)).Distinct().Count()==87,"duplicate call");
        return calls;
    }

    static int Main(string[] args)
    {
        Require(args.Length==5,"component fixtures reference result role");
        string role=args[4];Require(role is "current" or "candidate" or "verify","role");
        Require(Avx512F.IsSupported&&Avx2.IsSupported&&Fma.IsSupported&&Environment.Version.ToString()=="10.0.8","runtime/ISA");
        Require(FileHash(args[0])==ComponentHash,"qualified component identity");
        var assembly=Assembly.LoadFrom(args[0]);var type=assembly.GetType("Lokad.Onnx.ConvBlockedSpatial",true)!;
        var prepareDirect=Bind<Prepare>(type,"Prepare");var prepareWinograd=Bind<Prepare>(type,"PrepareWinograd");
        var direct=Bind<Direct>(type,"Execute");var winograd=Bind<Winograd>(type,"ExecuteWinograd");var plan=Bind<Plan>(type,"PlanWinograd");
        var tensors=new Dictionary<string,float[]>();var calls=Load(args[1],args[2],tensors);
        var hashes=tensors.ToDictionary(x=>x.Key,x=>Hash(x.Value));
        var constants=calls.DistinctBy(c=>c.Index).ToArray();Require(constants.Length==29,"prepared node census");
        var observations=new List<object>();var preparation=new List<object>();var held=new List<(float[],string)>();
        var preparedHashes=new Dictionary<(bool,int),string>();int verified=0;
        foreach (bool candidate in role=="verify"?new[]{false,true}:new[]{role=="candidate"})
        {
            string leg=candidate?"candidate":"current";var prepare=candidate?prepareWinograd:prepareDirect;
            for (int pass=0;pass<(role=="verify"?1:4);pass++)
            {
                var packed=new Dictionary<int,float[]>();long preparedBytes=0;
                foreach (var call in constants)
                {
                    long start=Stopwatch.GetTimestamp();
                    var weights=prepare(call.Weights,call.C,call.M,16);
                    long ticks=Stopwatch.GetTimestamp()-start;
                    Require(weights is not null,"finite prepared weights");
                    packed.Add(call.Index,weights);string hash=Hash(weights);
                    if (preparedHashes.TryGetValue((candidate,call.Index),out var prior)) Require(hash==prior,"preparation repeatability");
                    else preparedHashes.Add((candidate,call.Index),hash);
                    preparedBytes+=4L*weights.Length;
                    preparation.Add(new {kind="preparation",role=leg,pass,warmup=pass==0,index=call.Index,ticks,frequency=Stopwatch.Frequency,bytes=4L*weights.Length,sha256=hash});
                }
                Require(preparedBytes<=64L*1024*1024,"prepared budget");
                foreach (var call in calls)
                {
                    string expected=candidate?call.CandidateHash:call.CurrentHash;
                    for (int iteration=0;iteration<(role=="verify"?1:call.Iterations);iteration++)
                    {
                        long start=Stopwatch.GetTimestamp();
                        var output=CompleteCall(call,packed[call.Index],candidate,direct,winograd,plan,out long requested,out long rented);
                        long ticks=Stopwatch.GetTimestamp()-start;
                        string hash=Hash(output);Require(hash==expected,"qualified output mismatch");verified++;
                        if (pass==0&&iteration==0) held.Add((output,hash));
                        observations.Add(new {kind="call",role=leg,pass,warmup=pass==0,fixture=call.Fixture,index=call.Index,form=call.Form,
                            iteration,iterations=call.Iterations,work=call.Work,ticks,frequency=Stopwatch.Frequency,values=output.Length,
                            sha256=hash,exact=true,scratch_requested_bytes=requested,scratch_rented_bytes=rented,prepared_bytes=preparedBytes});
                    }
                }
                foreach (var pair in held) Require(Hash(pair.Item1)==pair.Item2,"held output mutated");
                foreach (var pair in packed) Require(Hash(pair.Value)==preparedHashes[(candidate,pair.Key)],"prepared weights mutated");
            }
        }
        Require(tensors.All(p=>Hash(p.Value)==hashes[p.Key]),"read-only operands");
        Require(verified==(role=="verify"?174:1044),"complete-call census");
        File.WriteAllText(args[3],JsonSerializer.Serialize(new {passed=true,role,protocol="winograd-87-geometry-2pow31-v1",component=ComponentHash,
            consumer=FileHash(typeof(Screen).Assembly.Location),pid=Environment.ProcessId,runtime=Environment.Version.ToString(),
            cases=87,calls=verified,warmups=role=="verify"?174:261,measured=role=="verify"?0:783,
            observations,preparation,read_only_operands=true,held_outputs=true},new JsonSerializerOptions {WriteIndented=true})+"\n");
        return 0;
    }
}
