using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using WidthProbe;

static string Hash(ReadOnlySpan<float> values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values)));
static void Require(bool value,string message) { if(!value)throw new Exception(message); }
int packingChecks=Layout.Prove();
if(args.Length==1 && args[0]=="--host")
{
    bool unsupported=!Avx512F.IsSupported||!Fma.IsSupported;
    if(unsupported)unsafe { Require(!Kernels.TryPackedAvx512Rows(128,384,384,null,null,null),"Unsupported host accepted"); Require(!InputPacked.TryPackedAvx512Rows(128,384,384,null,null,null),"Input-packed unsupported host accepted"); }
    Console.WriteLine(JsonSerializer.Serialize(new{Avx512F=Avx512F.IsSupported,Fma=Fma.IsSupported,unsupported_refusal_checked=unsupported,packing_checks=packingChecks}));return 0;
}
Require(Avx512F.IsSupported&&Fma.IsSupported,"Target AVX-512 host required");
if(args.Length==1 && args[0]=="--codegen")
{
    var a=Enumerable.Repeat(1f,24*384).ToArray();var p=Enumerable.Repeat(.25f,384*384).ToArray();var c=new float[24*384];
    unsafe{fixed(float* ap=a,pp=p,cp=c){Require(Kernels.TryPackedAvx512Rows(24,384,384,ap,pp,cp),"Original codegen");Require(InputPacked.TryPackedAvx512Rows(24,384,384,ap,pp,cp),"Candidate codegen");}}
    Require(c.All(v=>v==192f),"Codegen output mismatch");Console.WriteLine("Codegen-only outputs passed");return 0;
}
Require(Environment.ProcessorCount==1,"Affinity before CLR required");
string output=Path.GetFullPath(args[0]);int order=int.Parse(args[1]);Require(order>=1 && order<=4 && !File.Exists(output),"Invalid order or existing output");
int checkedCases=0;var checks=new List<object>(256);var records=new List<object>(36);var bankRecords=new List<object>(12);
unsafe void Invoke(int m,int n,int k,float* a,float* p,float* c,int mode)
{
    Require(mode<2?Kernels.TryPackedAvx512Rows(m,n,k,a,p,c):InputPacked.TryPackedAvx512Rows(m,n,k,a,p,c),"Kernel declined");
}
unsafe void Check(int m,int n,int k,bool exceptional)
{
    const int guard=5;const float sentinel=-12345.5f;var random=new Random(m*123+n+k);
    var a=Enumerable.Repeat(sentinel,m*n+2*guard).ToArray();var b=new float[n*k];var p=Enumerable.Repeat(sentinel,n*k+2*guard).ToArray();var initial=Enumerable.Repeat(sentinel,m*k+2*guard).ToArray();
    for(int i=0;i<m*n;i++)a[guard+i]=random.NextSingle()*2-1;
    for(int i=0;i<b.Length;i++)b[i]=random.NextSingle()*2-1;
    for(int i=0;i<m*k;i++)initial[guard+i]=random.NextSingle()*2-1;
    if(exceptional){a[guard]=-0f;a[guard+n+1]=BitConverter.Int32BitsToSingle(0x7fa12345);b[17]=float.PositiveInfinity;b[^1]=float.NegativeInfinity;initial[guard+1]=-0f;}
    string ah=Hash(a),bh=Hash(b);var baseline=initial.ToArray();
    fixed(float* ap=a,bp=b,pp=p,cp=baseline)
    {
        Lokad.Onnx.MathOps.PackPanelsB(n,k,bp,pp+guard);string ph=Hash(p);
        Require(p.Take(guard).Concat(p.Skip(guard+n*k)).All(v=>v==sentinel),"Pack guards");
        Invoke(m,n,k,ap+guard,pp+guard,cp+guard,0);
        Require(baseline.Take(guard).Concat(baseline.Skip(guard+m*k)).All(v=>v==sentinel),"Output guards");
        foreach(int mode in new[]{1,2})
        {
            var actual=initial.ToArray();fixed(float* target=actual)Invoke(m,n,k,ap+guard,pp+guard,target+guard,mode);
            Require(Hash(actual)==Hash(baseline),$"Bits {m},{n},{k},mode{mode},exceptional{exceptional}");
            for(int row=0;row<m;row++)foreach(int col in new[]{0,15,16,31,32,47,48,63,64,k-1})
            {
                float expected=initial[guard+row*k+col];
                for(int j=0;j<n;j++)expected=MathF.FusedMultiplyAdd(a[guard+row*n+j],b[j*k+col],expected);
                float v=actual[guard+row*k+col];Require(float.IsNaN(expected)?float.IsNaN(v):BitConverter.SingleToInt32Bits(expected)==BitConverter.SingleToInt32Bits(v),"Scalar FMA mismatch");
            }
            // Repeat accumulation with an already nonzero result; neither mode may overwrite it.
            fixed(float* target=actual)Invoke(m,n,k,ap+guard,pp+guard,target+guard,mode);
            var twice=baseline.ToArray();fixed(float* target=twice)Invoke(m,n,k,ap+guard,pp+guard,target+guard,0);
            Require(Hash(actual)==Hash(twice),"Second accumulation changed");
        }
        Require(ph==Hash(p),"Packed values changed");
        checks.Add(new{m,n,k,exceptional,a_sha256=ah,b_sha256=bh,packed_sha256=ph,output_sha256=Hash(baseline)});
    }
    Require(ah==Hash(a)&&bh==Hash(b),"Inputs changed");checkedCases++;
}
foreach(int m in Enumerable.Range(8,38).Concat(new[]{64,128,512}))
foreach(int n in new[]{1,65,129,257,385,513})Check(m,n,96,false);
foreach(int m in new[]{8,13,14,15,27,28,29,30,42,128})Check(m,385,192,true);
unsafe
{
    foreach(var (m,n,k) in new[]{(7,384,384),(8,0,384),(8,384,31),(8,384,0),(14,-1,32),(14,5,-32),(0,5,32),(14,5,33)})
        Require(!InputPacked.TryPackedAvx512Rows(m,n,k,null,null,null),"Invalid geometry accepted");
}
Console.WriteLine("Correctness cases: "+checkedCases);
int[] Modes(int index) => (order+index)%2==0?new[]{2,1,0}:new[]{0,1,2};
int shape=0;
foreach(int m in new[]{8,30,128,512})foreach(var (n,k) in new[]{(384,384),(384,1536),(1536,384)})
{
    var random=new Random(17+m+n+k);var a=Enumerable.Range(0,m*n).Select(_=>random.NextSingle()-.5f).ToArray();var b=Enumerable.Range(0,n*k).Select(_=>random.NextSingle()-.5f).ToArray();var p=new float[n*k];var c=new float[m*k];
    string ah=Hash(a),bh=Hash(b);string expected;var pack=Stopwatch.StartNew();
    unsafe{fixed(float* bp=b,pp=p)Lokad.Onnx.MathOps.PackPanelsB(n,k,bp,pp);}pack.Stop();
    unsafe{fixed(float* ap=a,pp=p,cp=c)Invoke(m,n,k,ap,pp,cp,0);}expected=Hash(c);string ph=Hash(p);
    foreach(int mode in Modes(shape++))
    {
        int iterations=m<64?64:m<256?16:4;var samples=new List<double>(7);var gcBefore=Enumerable.Range(0,3).Select(GC.CollectionCount).ToArray();long allocatedBefore=GC.GetTotalAllocatedBytes();
        unsafe{fixed(float* ap=a,pp=p,cp=c)
        {
            for(int warm=0;warm<8;warm++){Array.Clear(c);Invoke(m,n,k,ap,pp,cp,mode);}
            for(int sample=0;sample<7;sample++)
            {
                long start=Stopwatch.GetTimestamp();
                for(int i=0;i<iterations;i++){Array.Clear(c);Invoke(m,n,k,ap,pp,cp,mode);}
                samples.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds/iterations);
            }
        }}
        Require(Hash(c)==expected&&Hash(a)==ah&&Hash(b)==bh&&Hash(p)==ph,"Timed bits/inputs changed");
        records.Add(new{m,n,k,mode,iterations,gc_before=gcBefore,gc_after=Enumerable.Range(0,3).Select(GC.CollectionCount).ToArray(),allocated_bytes=GC.GetTotalAllocatedBytes()-allocatedBefore,samples_ms=samples,pack_ms=pack.Elapsed.TotalMilliseconds,packed_bytes=p.Length*4L,a_sha256=ah,b_sha256=bh,packed_sha256=ph,output_sha256=expected});
    }
}
// Each layer supplies Q, K, V, output, expansion and contraction prepared weights.
var weights=new List<(int N,int K,float[] Packed,string Hash)>();long bankBytes=0;
var packWatch=Stopwatch.StartNew();
for(int layer=0;layer<12;layer++)foreach(var (n,k) in new[]{(384,384),(384,384),(384,384),(384,384),(384,1536),(1536,384)})
{
    var random=new Random(789+weights.Count);var b=Enumerable.Range(0,n*k).Select(_=>random.NextSingle()-.5f).ToArray();var p=new float[n*k];
    unsafe{fixed(float* bp=b,pp=p)Lokad.Onnx.MathOps.PackPanelsB(n,k,bp,pp);}weights.Add((n,k,p,Hash(p)));bankBytes+=p.Length*4L;
}
packWatch.Stop();Require(bankBytes==84934656,"Weight bank geometry");
foreach(int m in new[]{8,30,128,512})
{
    var random=new Random(901+m);var inputs=new Dictionary<int,float[]>();var outputs=new Dictionary<int,float[]>();
    foreach(int d in new[]{384,1536}){inputs[d]=Enumerable.Range(0,m*d).Select(_=>random.NextSingle()-.5f).ToArray();outputs[d]=new float[m*d];}
    var inputHashes=inputs.ToDictionary(p=>p.Key,p=>Hash(p.Value));var hashes=new string[weights.Count];
    unsafe void Matrix(int index,int mode)
    {
        var w=weights[index];float[] c=outputs[w.K];Array.Clear(c);
        fixed(float* ap=inputs[w.N],pp=w.Packed,cp=c)Invoke(m,w.N,w.K,ap,pp,cp,mode);
    }
    for(int i=0;i<weights.Count;i++){Matrix(i,0);hashes[i]=Hash(outputs[weights[i].K]);}
    foreach(int mode in Modes(shape++))
    {
        var gcBefore=Enumerable.Range(0,3).Select(GC.CollectionCount).ToArray();long allocatedBefore=GC.GetTotalAllocatedBytes();
        for(int warm=0;warm<2;warm++)for(int i=0;i<weights.Count;i++)Matrix(i,mode);
        var samples=new List<double>(7);
        for(int sample=0;sample<7;sample++)
        {
            long start=Stopwatch.GetTimestamp();for(int i=0;i<weights.Count;i++)Matrix(i,mode);
            samples.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        }
        for(int i=0;i<weights.Count;i++){Matrix(i,mode);Require(Hash(outputs[weights[i].K])==hashes[i],"Bank output changed");}
        Require(weights.All(w=>Hash(w.Packed)==w.Hash)&&inputs.All(p=>Hash(p.Value)==inputHashes[p.Key]),"Bank inputs changed");
        bankRecords.Add(new{m,mode,gc_before=gcBefore,gc_after=Enumerable.Range(0,3).Select(GC.CollectionCount).ToArray(),allocated_bytes=GC.GetTotalAllocatedBytes()-allocatedBefore,matrices=weights.Count,packed_bytes=bankBytes,samples_ms=samples,output_sha256=hashes});
        Console.WriteLine($"Bank m={m} mode={mode} median={samples.Order().ElementAt(3):F4}");
    }
}
using var process=Process.GetCurrentProcess();
File.WriteAllText(output,JsonSerializer.Serialize(new{checked_cases=checkedCases,checks,packing_checks=packingChecks,refusals=8,order,flags=Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k=>k.StartsWith("LOKAD_")||k.StartsWith("DOTNET_")||k.StartsWith("COMPlus_")).ToDictionary(k=>k,Environment.GetEnvironmentVariable),runtime=Environment.Version.ToString(),processor_count=Environment.ProcessorCount,
    affinity=OperatingSystem.IsLinux()||OperatingSystem.IsWindows()?process.ProcessorAffinity.ToInt64():0,avx512=Avx512F.IsSupported,core_sha256=Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(Lokad.Onnx.MathOps).Assembly.Location))),
    producer_sha256=Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(Kernels).Assembly.Location))),peak=process.PeakWorkingSet64,results=records,bank_setup_ms=packWatch.Elapsed.TotalMilliseconds,bank=bankRecords},new JsonSerializerOptions{WriteIndented=true}));return 0;

static class Layout
{
    public static unsafe int Prove()
    {
        int cases=0;
        foreach(int rows in new[]{0,12,24,36,120,504})
        foreach(int reduction in new[]{1,2,7,65,129,257,384,513,1536})
        {
            const int guard=5;int count=rows*reduction;
            var source=new float[count+guard*2];var packed=new float[count+guard*2];
            for(int i=0;i<source.Length;i++)source[i]=BitConverter.Int32BitsToSingle(unchecked(i*1664525+1013904223));
            Array.Fill(packed,-12345.5f);
            var before=MemoryMarshal.AsBytes(source.AsSpan()).ToArray();
            fixed(float* a=source,p=packed)InputPacked.PackRows12(rows,reduction,a+guard,p+guard);
            for(int i=0;i<rows;i+=12)for(int j=0;j<reduction;j++)for(int r=0;r<12;r++)
                if(BitConverter.SingleToInt32Bits(packed[guard+i*reduction+j*12+r])!=BitConverter.SingleToInt32Bits(source[guard+(i+r)*reduction+j]))throw new InvalidDataException("Packing bits changed");
            if(!before.AsSpan().SequenceEqual(MemoryMarshal.AsBytes(source.AsSpan())))throw new InvalidDataException("Packing input mutated");
            if(packed.Take(guard).Concat(packed.Skip(guard+count)).Any(v=>v!=-12345.5f))throw new InvalidDataException("Packing guard changed");
            cases++;
        }
        return cases;
    }
}
