using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using WidthProbe;

if(args.Length>0 && args[0]=="--trace-smoke")
{
    Measurement.Gate(args[1]);
    for(int i=0;i<10;i++){var timing=Measurement.Begin("smoke",1,1,1,0,i,"measured",10000);for(int j=0;j<10000;j++)Measurement.Smoke(j);Measurement.End(timing,Stopwatch.GetTimestamp());Thread.Sleep(30);}
    File.WriteAllText(args[2],JsonSerializer.Serialize(Measurement.Rows));return 0;
}
if (args.Length == 1 && args[0] == "--host")
{
    bool unsupported = !Avx512F.IsSupported || !Fma.IsSupported;
    if (unsupported) unsafe {
        Proof.Require(!Kernels.TryPackedAvx512Rows(128,384,384,null,null,null), "Unsupported original accepted");
        Proof.Require(!GeneratedControl.TryPackedAvx512Rows(128,384,384,null,null,null), "Unsupported control accepted");
        Proof.Require(!GeneratedOverwrite.TryPackedAvx512Rows(128,384,384,null,null,null), "Unsupported overwrite accepted");
    }
    // Exercise the allocation/offset/lifetime contract even on a non-AVX512 host.
    using var probe = new PackSet(65,96,Enumerable.Range(0,65*96).Select(i=>(float)i/79).ToArray());
    probe.Verify();
    Console.WriteLine(JsonSerializer.Serialize(new { avx512=Avx512F.IsSupported, fma=Fma.IsSupported, unsupported_refusal_checked=unsupported, buffers=probe.Info() }));
    return 0;
}
Proof.Require(Avx512F.IsSupported && Fma.IsSupported, "Target AVX512 host required");
Proof.Require(Environment.ProcessorCount == 1, "Affinity before CLR required");
Measurement.Gate(args[2]);
bool codegen=false;
string output = codegen?"":Path.GetFullPath(args[0]); int order = codegen?0:int.Parse(args[1]);
Proof.Require(order >= 0 && order < 10 && !File.Exists(output), "Invalid order or existing output");
var checks = new List<object>(); var records = new List<object>(); var bankRecords = new List<object>();
unsafe void Invoke(int m,int n,int k,float* a,Panel p,float* c,int mode)
{
    bool ok=mode<2 ? Kernels.TryPackedAvx512Rows(m,n,k,a,p.Pointer,c) : mode==2
        ? GeneratedControl.TryPackedAvx512Rows(m,n,k,a,p.Pointer,c)
        : GeneratedOverwrite.TryPackedAvx512Rows(m,n,k,a,p.Pointer,c);
    Proof.Require(ok, "Kernel declined");
}
unsafe void Check(int m,int n,int k,bool exceptional)
{
    const int guard=5;var random=new Random(m*123+n+k);
    var a=Enumerable.Repeat(Proof.Sentinel,m*n+2*guard).ToArray();var b=new float[n*k];
    var initial=Enumerable.Repeat(Proof.Sentinel,m*k+2*guard).ToArray();
    for(int i=0;i<m*n;i++)a[guard+i]=random.NextSingle()*2-1;
    for(int i=0;i<b.Length;i++)b[i]=random.NextSingle()*2-1;
    for(int i=0;i<m*k;i++)initial[guard+i]=random.NextSingle()*2-1;
    if(exceptional){a[guard]=-0f;a[guard+n+1]=BitConverter.Int32BitsToSingle(0x7fa12345);b[17]=float.PositiveInfinity;b[^1]=float.NegativeInfinity;initial[guard+1]=-0f;initial[guard+3]=BitConverter.Int32BitsToSingle(unchecked((int)0xffa54321));}
    string ah=Proof.Hash(a),bh=Proof.Hash(b);using var packed=new PackSet(n,k,b);
    var baseline=initial.ToArray();Array.Clear(baseline,guard,m*k);
    fixed(float* ap=a,cp=baseline)Invoke(m,n,k,ap+guard,packed.Modes[0],cp+guard,0);
    var accumulated=initial.ToArray();fixed(float* ap=a,cp=accumulated)Invoke(m,n,k,ap+guard,packed.Modes[0],cp+guard,0);
    var twice=accumulated.ToArray();fixed(float* ap=a,cp=twice)Invoke(m,n,k,ap+guard,packed.Modes[0],cp+guard,0);
    foreach(int mode in Enumerable.Range(0,5))
    {
        var actual=initial.ToArray();if(mode<4)Array.Clear(actual,guard,m*k);
        fixed(float* ap=a,cp=actual)Invoke(m,n,k,ap+guard,packed.Modes[mode],cp+guard,mode);
        Proof.Require(Proof.Hash(actual)==Proof.Hash(baseline),$"Overwrite bits {m},{n},{k},mode{mode},exceptional{exceptional}");
        Proof.Require(actual.Take(guard).Concat(actual.Skip(guard+m*k)).All(v=>v==Proof.Sentinel),"Output guards");
        for(int row=0;row<m;row++)foreach(int col in new[]{0,15,16,31,32,47,48,63,64,k-1})
        {
            float expected=0f;for(int j=0;j<n;j++)expected=MathF.FusedMultiplyAdd(a[guard+row*n+j],b[j*k+col],expected);
            float value=actual[guard+row*k+col];
            Proof.Require(float.IsNaN(expected)?float.IsNaN(value):BitConverter.SingleToInt32Bits(expected)==BitConverter.SingleToInt32Bits(value),"Scalar FMA mismatch");
        }
        if(mode>=3)
        {
            fixed(float* ap=a,cp=actual)Invoke(m,n,k,ap+guard,packed.Modes[mode],cp+guard,mode);
            Proof.Require(Proof.Hash(actual)==Proof.Hash(baseline),"Second overwrite changed");
        }
        else
        {
            actual=initial.ToArray();fixed(float* ap=a,cp=actual)Invoke(m,n,k,ap+guard,packed.Modes[mode],cp+guard,mode);
            Proof.Require(Proof.Hash(actual)==Proof.Hash(accumulated),"Dirty accumulation changed");
            fixed(float* ap=a,cp=actual)Invoke(m,n,k,ap+guard,packed.Modes[mode],cp+guard,mode);
            Proof.Require(Proof.Hash(actual)==Proof.Hash(twice),"Second accumulation changed");
        }
        packed.Verify();
    }
    Proof.Require(Proof.Hash(a)==ah && Proof.Hash(b)==bh,"Source input mutation");
    checks.Add(new {m,n,k,exceptional,a_sha256=ah,b_sha256=bh,output_sha256=Proof.Hash(baseline),accumulated_sha256=Proof.Hash(accumulated),twice_sha256=Proof.Hash(twice),buffers=packed.Info()});
}
foreach(int m in Enumerable.Range(8,38).Concat(new[]{64,128,512}))
foreach(int n in new[]{1,65,129,257,385,513})Check(m,n,96,false);
foreach(int m in new[]{8,13,14,15,27,28,29,30,42,128})Check(m,385,192,true);
unsafe
{
    var sentinels=Enumerable.Repeat(Proof.Sentinel,32).ToArray();string before=Proof.Hash(sentinels);
    fixed(float* p=sentinels)
    foreach(var (m,n,k) in new[]{(7,384,384),(8,0,384),(8,384,31),(8,384,0),(14,-1,32),(14,5,-32),(0,5,32),(14,5,33)})
    {
        Proof.Require(!Kernels.TryPackedAvx512Rows(m,n,k,p,p,p),"Invalid original geometry accepted");
        Proof.Require(!GeneratedControl.TryPackedAvx512Rows(m,n,k,p,p,p),"Invalid control geometry accepted");
        Proof.Require(!GeneratedOverwrite.TryPackedAvx512Rows(m,n,k,p,p,p),"Invalid overwrite geometry accepted");
    }
    Proof.Require(Proof.Hash(sentinels)==before,"Refusal mutated data");
}
Console.WriteLine("Correctness cases: "+checks.Count);
if(codegen)return 0;
int[] Modes(int shape)
{
    var modes=Enumerable.Range(0,5).Select(i=>(i+order%5+shape)%5).ToArray();
    if(order>=5)Array.Reverse(modes);
    return modes;
}
int shape=0;
foreach(int m in new[]{8,30,128,512})foreach(var (n,k) in new[]{(384,384),(384,1536),(1536,384)})
{
    var random=new Random(17+m+n+k);
    var a=Enumerable.Range(0,m*n).Select(_=>random.NextSingle()-.5f).ToArray();
    var b=Enumerable.Range(0,n*k).Select(_=>random.NextSingle()-.5f).ToArray(); var c=new float[m*k];
    string ah=Proof.Hash(a),bh=Proof.Hash(b); using var packed=new PackSet(n,k,b);
    unsafe { fixed(float* ap=a,cp=c)Invoke(m,n,k,ap,packed.Modes[0],cp,0); }
    string expected=Proof.Hash(c); int shapeIndex=shape++;
    foreach(int mode in Modes(shapeIndex))
    {
        int iterations=m<64?64:m<256?16:4; var samples=new List<double>(7); var gcBefore=Proof.Gc(); long allocatedBefore=GC.GetTotalAllocatedBytes();
        var panel=packed.Modes[mode];panel.Verify();
        unsafe { fixed(float* ap=a,cp=c)
        {
            var warmTiming=Measurement.Begin("isolated",m,n,k,mode,-1,"warmup",8);
            for(int warm=0;warm<8;warm++){if(mode<4)Array.Clear(c);Invoke(m,n,k,ap,panel,cp,mode);}
            Measurement.End(warmTiming,Stopwatch.GetTimestamp());
            for(int sample=0;sample<7;sample++)
            {
                var timing=Measurement.Begin("isolated",m,n,k,mode,sample,"measured",iterations);
                for(int i=0;i<iterations;i++){if(mode<4)Array.Clear(c);Invoke(m,n,k,ap,panel,cp,mode);}
                samples.Add(Measurement.End(timing,Stopwatch.GetTimestamp())/iterations);
            }
        }}
        var gcAfter=Proof.Gc();long allocatedAfter=GC.GetTotalAllocatedBytes();
        Proof.Require(Proof.Hash(c)==expected && Proof.Hash(a)==ah && Proof.Hash(b)==bh, "Timed bits/inputs changed");packed.Verify();
        records.Add(new {m,n,k,shape=shapeIndex,mode,sequence=Modes(shapeIndex),iterations,samples_ms=samples,buffer=panel.Info(),gc_before=gcBefore,gc_after=gcAfter,allocated_bytes=allocatedAfter-allocatedBefore,a_sha256=ah,b_sha256=bh,output_sha256=expected});
    }
}
// Twelve layers, four width projections, one expansion, one contraction per layer.
var weights=new List<PackSet>();
try
{
    for(int layer=0;layer<12;layer++)foreach(var (n,k) in new[]{(384,384),(384,384),(384,384),(384,384),(384,1536),(1536,384)})
    {
        var random=new Random(789+weights.Count);var b=Enumerable.Range(0,n*k).Select(_=>random.NextSingle()-.5f).ToArray();
        weights.Add(new PackSet(n,k,b));
    }
    long bankBytes=weights.Sum(w=>w.N*(long)w.K*4);Proof.Require(bankBytes==84934656, "Bank geometry");
    foreach(int m in new[]{8,30,128,512})
    {
        var random=new Random(901+m);var inputs=new Dictionary<int,float[]>();var outputs=new Dictionary<int,float[]>();
        foreach(int d in new[]{384,1536}){inputs[d]=Enumerable.Range(0,m*d).Select(_=>random.NextSingle()-.5f).ToArray();outputs[d]=new float[m*d];}
        var inputHashes=inputs.ToDictionary(p=>p.Key,p=>Proof.Hash(p.Value));var hashes=new string[weights.Count];
        unsafe void Matrix(int index,int mode)
        {
            var w=weights[index];var c=outputs[w.K];if(mode<4)Array.Clear(c);
            fixed(float* ap=inputs[w.N],cp=c)Invoke(m,w.N,w.K,ap,w.Modes[mode],cp,mode);
        }
        for(int i=0;i<weights.Count;i++){Matrix(i,0);hashes[i]=Proof.Hash(outputs[weights[i].K]);}
        int shapeIndex=shape++;
        foreach(int mode in Modes(shapeIndex))
        {
            foreach(var w in weights)w.Verify();
            var gcBefore=Proof.Gc();long allocatedBefore=GC.GetTotalAllocatedBytes();
            var warmTiming=Measurement.Begin("bank",m,0,0,mode,-1,"warmup",2);
            for(int warm=0;warm<2;warm++)for(int i=0;i<weights.Count;i++)Matrix(i,mode);
            Measurement.End(warmTiming,Stopwatch.GetTimestamp());
            var samples=new List<double>(7);
            for(int sample=0;sample<7;sample++)
            {
                var timing=Measurement.Begin("bank",m,0,0,mode,sample,"measured",1);
                for(int i=0;i<weights.Count;i++)Matrix(i,mode);
                samples.Add(Measurement.End(timing,Stopwatch.GetTimestamp()));
            }
            var gcAfter=Proof.Gc();long allocatedAfter=GC.GetTotalAllocatedBytes();
            for(int i=0;i<weights.Count;i++){Matrix(i,mode);Proof.Require(Proof.Hash(outputs[weights[i].K])==hashes[i], "Bank output changed");}
            Proof.Require(inputs.All(p=>Proof.Hash(p.Value)==inputHashes[p.Key]), "Bank input mutation");
            foreach(var w in weights)w.Verify();
            bankRecords.Add(new {m,shape=shapeIndex,mode,sequence=Modes(shapeIndex),matrices=weights.Count,packed_bytes=bankBytes,samples_ms=samples,output_sha256=hashes,gc_before=gcBefore,gc_after=gcAfter,allocated_bytes=allocatedAfter-allocatedBefore,buffers=weights.Select(w=>w.Modes[mode].Info()).ToArray()});
            Console.WriteLine($"Bank m={m} mode={mode} median={samples.Order().ElementAt(3):F4}");
        }
    }
    using var process=Process.GetCurrentProcess();
    File.WriteAllText(output,JsonSerializer.Serialize(new {checked_cases=checks.Count,checks,refusals=8,order,runtime=Environment.Version.ToString(),processor_count=Environment.ProcessorCount,
        affinity=(OperatingSystem.IsLinux()||OperatingSystem.IsWindows())?process.ProcessorAffinity.ToInt64():0,avx512=Avx512F.IsSupported,
        core_sha256=Proof.FileHash(typeof(Lokad.Onnx.MathOps).Assembly.Location),producer_sha256=Proof.FileHash(typeof(Panel).Assembly.Location),peak=process.PeakWorkingSet64,
        results=records,bank=bankRecords,intervals=Measurement.Rows,stopwatch_frequency=Stopwatch.Frequency,process_id=Environment.ProcessId,trace_enabled=Markers.Log.IsEnabled(),flags=Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k=>k.StartsWith("LOKAD_")||k.StartsWith("DOTNET_")||k.StartsWith("COMPlus_")).ToDictionary(k=>k,Environment.GetEnvironmentVariable)},new JsonSerializerOptions{WriteIndented=true}));
}
finally { foreach(var w in weights)w.Dispose(); }
return 0;

static class Proof
{
    public const float Sentinel=-12345.5f;
    public static void Require(bool value,string message){if(!value)throw new InvalidDataException(message);}
    public static string Hash(ReadOnlySpan<float> values)=>Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values)));
    public static string FileHash(string path)=>Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
    public static int[] Gc()=>Enumerable.Range(0,3).Select(GC.CollectionCount).ToArray();
}
unsafe sealed class Panel:IDisposable
{
    readonly float[] array; readonly GCHandle handle; readonly ulong address; readonly string physicalHash,logicalHash;
    readonly int length,offset;readonly int? target;readonly double allocationMs,packMs;
    public float* Pointer => (float*)handle.AddrOfPinnedObject()+offset;
    public Panel(int n,int k,float[] source,int? residue)
    {
        length=checked(n*k);target=residue;var watch=Stopwatch.StartNew();
        array=new float[length+(target.HasValue?31:0)];Array.Fill(array,Proof.Sentinel);
        handle=GCHandle.Alloc(array,GCHandleType.Pinned);
        if(target.HasValue){ulong start=(ulong)handle.AddrOfPinnedObject()+32;offset=8+(int)(((ulong)target.Value+64-start%64)%64)/4;}
        address=(ulong)Pointer;watch.Stop();allocationMs=watch.Elapsed.TotalMilliseconds;
        watch.Restart();fixed(float* sourcePointer=source)Lokad.Onnx.MathOps.PackPanelsB(n,k,sourcePointer,Pointer);watch.Stop();packMs=watch.Elapsed.TotalMilliseconds;
        logicalHash=Proof.Hash(array.AsSpan(offset,length));physicalHash=Proof.Hash(array);Verify();
    }
    public void Verify()
    {
        Proof.Require((ulong)Pointer==address && (!target.HasValue || address%64==(ulong)target.Value), "Address changed");
        Proof.Require(Proof.Hash(array)==physicalHash && Proof.Hash(array.AsSpan(offset,length))==logicalHash, "Packed mutation");
        Proof.Require(array.AsSpan(0,offset).IndexOfAnyExcept(Proof.Sentinel)<0 && array.AsSpan(offset+length).IndexOfAnyExcept(Proof.Sentinel)<0, "Packed guards");
        Proof.Require(!target.HasValue || (offset>=8 && array.Length-offset-length>=8), "Guard geometry");
    }
    public object Info()=>new {address,mod64=address%64,target,offset,logical_bytes=length*4L,array_bytes=array.LongLength*4,logical_sha256=logicalHash,physical_sha256=physicalHash,allocation_ms=allocationMs,pack_ms=packMs};
    public void Dispose(){if(handle.IsAllocated)handle.Free();}
}
sealed class PackSet:IDisposable
{
    public int N{get;} public int K{get;} public Panel[] Modes{get;}
    public PackSet(int n,int k,float[] source)
    {
        N=n;K=k;var natural=new Panel(n,k,source,null);
        Modes=new[]{natural,natural,natural,natural,natural};
        // Every mode must share this exact pinned panel. Verify the typed storage
        // contract directly; JSON here used to queue compilation into later timings.
        Verify();
    }
    public void Verify(){foreach(var p in Modes.Distinct())p.Verify();Proof.Require(Modes.All(p=>ReferenceEquals(p,Modes[0])), "Shared buffer changed");}
    public object[] Info()=>Modes.Select(p=>p.Info()).ToArray();
    public void Dispose(){foreach(var p in Modes.Distinct())p.Dispose();}
}
