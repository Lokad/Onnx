using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;

static class Benchmark
{
    static readonly string[] Names={"e5-8tok","e5-30tok","e5-30pad128","e5-128tok","e5-512tok"};
    static readonly int[] Repeats={64,32,8,8,2};
    static readonly string[] Variants={"Product","CopyA","CopyB","Conditional"};
    static readonly Kernel[] Methods={Kernels.Product,Kernels.CopyA,Kernels.CopyB,Kernels.Conditional};
    sealed record Bank(float[][] X,float[][] Bias,float[][] Y,float[][] Expected,string InputHash,string BiasHash,string ExpectedHash);
    record Sample(int cycle,int position,string variant,int repeats,long ticks,long allocated,int[] gc,string output_sha256);
    static string Hash(float[][] values)
    {
        using var hash=IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach(var v in values)hash.AppendData(MemoryMarshal.AsBytes(v.AsSpan()));
        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }
    [MethodImpl(MethodImplOptions.AggressiveOptimization|MethodImplOptions.NoInlining)]
    static (long ticks,long allocated,int g0,int g1,int g2) Measure(Kernel kernel,Bank bank,int repeats)
    {
        int g0=GC.CollectionCount(0),g1=GC.CollectionCount(1),g2=GC.CollectionCount(2);
        long allocated=GC.GetAllocatedBytesForCurrentThread(),start=Stopwatch.GetTimestamp();
        for(int i=0;i<repeats;i++)for(int layer=0;layer<12;layer++)kernel(bank.X[layer],bank.Bias[layer],bank.Y[layer]);
        long end=Stopwatch.GetTimestamp();
        return(end-start,GC.GetAllocatedBytesForCurrentThread()-allocated,GC.CollectionCount(0)-g0,GC.CollectionCount(1)-g1,GC.CollectionCount(2)-g2);
    }
    public static void Run(string capture,string output,int visit)
    {
        if(!OperatingSystem.IsLinux() || !Avx512F.IsSupported || Environment.ProcessorCount!=1 || Process.GetCurrentProcess().ProcessorAffinity.ToInt64()!=4 || visit<0 || visit>3 || File.Exists(output))throw new InvalidDataException("Wrong timing host/visit/destination");
        var flags=Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k=>k.StartsWith("LOKAD_",StringComparison.Ordinal)||k.StartsWith("DOTNET_",StringComparison.Ordinal)||k.StartsWith("COMPlus_",StringComparison.Ordinal)).ToArray();
        if(flags.Length!=0)throw new InvalidDataException("Timing overrides present");
        var banks=new Bank[5];
        using var bundle=JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Path.GetDirectoryName(capture)!,"bundle.json")));
        for(int index=0;index<5;index++)
        {
            var xs=new float[12][];var biases=new float[12][];var ys=new float[12][];var expected=new float[12][];
            for(int layer=0;layer<12;layer++)
            {
                string prefix=Path.Combine(capture,"capture",Names[index],layer.ToString("D2"));
                xs[layer]=MemoryMarshal.Cast<byte,float>(File.ReadAllBytes(prefix+"-x.f32")).ToArray();biases[layer]=MemoryMarshal.Cast<byte,float>(File.ReadAllBytes(prefix+"-bias.f32")).ToArray();
                ys[layer]=new float[xs[layer].Length];expected[layer]=new float[xs[layer].Length];Kernels.Product(xs[layer],biases[layer],expected[layer]);
            }
            banks[index]=new Bank(xs,biases,ys,expected,Hash(xs),Hash(biases),Hash(expected));
            var pinned=bundle.RootElement.GetProperty("banks").GetProperty(Names[index]);
            if(banks[index].InputHash!=pinned.GetProperty("input_sha256").GetString() || banks[index].BiasHash!=pinned.GetProperty("bias_sha256").GetString() || banks[index].ExpectedHash!=pinned.GetProperty("output_sha256").GetString())throw new InvalidDataException("Closed census bank hashes differ");
        }
        int[] caseOrder=Enumerable.Range(0,5).Select(i=>(i+visit)%5).ToArray();if(visit%2==1)Array.Reverse(caseOrder);
        var results=new object[5];
        foreach(int index in caseOrder)
        {
            var bank=banks[index];var warmup=new Sample[64];var measured=new Sample[192];
            for(int phase=0;phase<2;phase++)
            {
                int cycles=phase==0?16:48;
                for(int cycle=0;cycle<cycles;cycle++)for(int position=0;position<4;position++)
                {
                    int variant=(visit+cycle+position)%4;
                    var r=Measure(Methods[variant],bank,Repeats[index]);string outputHash=Hash(bank.Y);
                    if(outputHash!=bank.ExpectedHash)throw new InvalidDataException("Timed output differs: "+Names[index]+"/"+Variants[variant]);
                    var sample=new Sample(cycle,position,Variants[variant],Repeats[index],r.ticks,r.allocated,new[]{r.g0,r.g1,r.g2},outputHash);
                    (phase==0?warmup:measured)[cycle*4+position]=sample;
                }
            }
            if(Hash(bank.X)!=bank.InputHash || Hash(bank.Bias)!=bank.BiasHash || Hash(bank.Expected)!=bank.ExpectedHash)throw new InvalidDataException("Held input/bias/reference changed");
            results[index]=new{name=Names[index],layers=12,values=bank.X.Sum(x=>x.Length),input_sha256=bank.InputHash,bias_sha256=bank.BiasHash,output_sha256=bank.ExpectedHash,held_unchanged=true,warmup,measured};
            Console.WriteLine("Timed complete bank "+Names[index]);
        }
        using var stream=new FileStream(output,FileMode.CreateNew);
        JsonSerializer.Serialize(stream,new{schema=1,protocol="uniform-gelu-bank-v1",visit,frequency=Stopwatch.Frequency,runtime=RuntimeInformation.FrameworkDescription,affinity=4,avx512=Avx512F.IsSupported,flags,case_order=caseOrder.Select(i=>Names[i]).ToArray(),results},new JsonSerializerOptions{WriteIndented=true});
    }
}
