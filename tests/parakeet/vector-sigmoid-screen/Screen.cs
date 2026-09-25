using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

static class Screen
{
    static void Require(bool condition, string message) { if (!condition) throw new InvalidOperationException(message); }
    static string FileHash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
    static string Hash(ITensor value) => value switch
    {
        Tensor<float> f => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(f.ToArray().AsSpan()))),
        Tensor<double> d => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(d.ToArray().AsSpan()))),
        _ => throw new InvalidOperationException("dtype")
    };
    sealed class Fixture
    {
        public required string Name, Kind, Dtype;
        public required int[] Shape;
        public required int Index, Weight, Batch;
        public required ITensor Source, Parent;
        public required double[] Expected;
        public required string InputHash, ParentHash;
        public required OpResult[] Returned;
        public ExecutionOptions? Options;
        public ITensor? Held, Last;
        public string? HeldHash;
        public double MaximumError;
        public long SetupTicks;
        public readonly List<Clock> Clocks = new(780);
    }
    record Clock(int iteration, bool warmup, long start, long ticks);

    static Fixture Prepare(JsonElement item)
    {
        long start = Stopwatch.GetTimestamp();
        int[] shape = item.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        string kind = item.GetProperty("kind").GetString()!, dtype = item.GetProperty("dtype").GetString()!;
        int length = item.GetProperty("elements").GetInt32(), batch = item.GetProperty("batch").GetInt32();
        static float Value(int i) => (i % 257 - 128) / 16f;
        ITensor source, parent;
        if (dtype == "double") source = parent = new DenseTensor<double>(Enumerable.Range(0,length).Select(i => (double)Value(i)).ToArray(), shape);
        else if (kind == "sliced")
        {
            var owner = new DenseTensor<float>(Enumerable.Range(0,33*34).Select(Value).ToArray(), new[] {33,34});
            parent = owner; source = new TensorSlice<float>(owner, new[] {new SliceIndex(0,33),new SliceIndex(0,34,2)});
        }
        else if (kind == "broadcast")
        {
            var owner = new DenseTensor<float>(Enumerable.Range(0,17).Select(Value).ToArray(), new[] {1,17});
            parent = owner; source = owner.BroadcastDim(0,33);
        }
        else if (kind == "reversed")
        {
            var owner = new DenseTensor<float>(shape,true);
            for (int i=0;i<33;i++) for (int j=0;j<17;j++) owner[i,j]=Value(i*17+j);
            source=parent=owner;
        }
        else source=parent=new DenseTensor<float>(Enumerable.Range(0,length).Select(Value).ToArray(),shape);
        double[] expected=source switch
        {
            Tensor<float> f => f.ToArray().Select(x => (double)(1f/(1f+MathF.Exp(-x)))).ToArray(),
            Tensor<double> d => d.ToArray().Select(x => 1.0/(1.0+Math.Exp(-x))).ToArray(),
            _ => throw new InvalidOperationException("dtype")
        };
        var fixture=new Fixture { Name=item.GetProperty("name").GetString()!,Kind=kind,Dtype=dtype,Shape=shape,
            Index=item.GetProperty("index").GetInt32(),Weight=item.GetProperty("weight").GetInt32(),Batch=batch,
            Source=source,Parent=parent,Expected=expected,InputHash=Hash(source),ParentHash=Hash(parent),
            Returned=new OpResult[batch],Options=kind=="scalar-option"?ExecutionOptions.Scalar:null };
        fixture.SetupTicks=Stopwatch.GetTimestamp()-start;
        return fixture;
    }

    static void VerifyOutput(Fixture fixture,ITensor tensor)
    {
        Require(tensor.Dims.SequenceEqual(fixture.Shape),"shape "+fixture.Name);
        Require(tensor.Length==fixture.Expected.Length,"length");
        double tolerance=fixture.Dtype=="float"?1e-6:1e-12;
        for(int i=0;i<fixture.Expected.Length;i++)
        {
            double actual=tensor is DenseTensor<float> f?f.Buffer.Span[i]:((DenseTensor<double>)tensor).Buffer.Span[i];
            double error=Math.Abs(actual-fixture.Expected[i]);
            Require(double.IsFinite(actual) && error<=tolerance,"numerical "+fixture.Name);
            fixture.MaximumError=Math.Max(fixture.MaximumError,error);
        }
    }

    static void Fill(ITensor tensor,double value)
    {
        if(tensor is DenseTensor<float> f)f.Buffer.Span.Fill((float)value);
        else if(tensor is DenseTensor<double> d)d.Buffer.Span.Fill(value);
        else throw new InvalidOperationException("owner layout");
    }

    static void Main(string[] args)
    {
        if (!OperatingSystem.IsLinux()) throw new PlatformNotSupportedException("AMD Linux screen");
        Require(args.Length==4,"base role sequence output");
        string folder=Path.GetFullPath(args[0]),role=args[1];int sequence=int.Parse(args[2]);
        Require(new[]{"current","candidate","candidate","current"}[sequence]==role,"order");
        using var process=Process.GetCurrentProcess();
        Require(Environment.Version.ToString()=="10.0.8" && Environment.ProcessorCount==1 && process.ProcessorAffinity==(nint)4,"runtime/affinity");
        Require(Vector.IsHardwareAccelerated,"SIMD");
        var flags=Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>()
            .Where(e=>new[]{"LOKAD_","DOTNET_","COMPlus_"}.Any(p=>((string)e.Key).StartsWith(p,StringComparison.OrdinalIgnoreCase)))
            .ToDictionary(e=>(string)e.Key,e=>(string)e.Value!);
        Require(flags.Count==0,"ordinary runtime flags");
        using var specification=JsonDocument.Parse(File.ReadAllText(Path.Combine(folder,"spec.json")));
        using var census=JsonDocument.Parse(File.ReadAllText(Path.Combine(folder,"census.json")));
        string core=FileHash(typeof(CPUExecutionProvider).Assembly.Location);
        Require(core==specification.RootElement.GetProperty("products").GetProperty(role).GetProperty("sha256").GetString(),"actual Core");
        var fixtures=census.RootElement.GetProperty("cases").EnumerateArray().Select(Prepare).ToArray();
        Require(fixtures.Length==46 && fixtures.Sum(f=>f.Weight)==1920,"census");
        // All cases complete 600 rounds before any case is measured.
        for(int iteration=0;iteration<780;iteration++)
        foreach(var fixture in fixtures)
        {
            long start=Stopwatch.GetTimestamp();
            for(int call=0;call<fixture.Batch;call++)
                fixture.Returned[call]=CPUExecutionProvider.Sigmoid(fixture.Source,fixture.Options);
            long ticks=Stopwatch.GetTimestamp()-start;
            Require(ticks>0,"clock");
            fixture.Clocks.Add(new(iteration,iteration<600,start,ticks));
            foreach(var result in fixture.Returned)Require(result.Status==OpStatus.Success && result.Outputs.Length==1,"status");
            fixture.Last=fixture.Returned[^1].Outputs[0];
            if(iteration==0)
            {
                fixture.Held=fixture.Returned[0].Outputs[0];fixture.HeldHash=Hash(fixture.Held);
            }
            if(iteration is 0 or 599 or 779)
                foreach(var result in fixture.Returned)VerifyOutput(fixture,result.Outputs[0]);
        }
        var rows=new List<object>();
        foreach(var fixture in fixtures)
        {
            Require(Hash(fixture.Source)==fixture.InputHash && Hash(fixture.Parent)==fixture.ParentHash,"input ownership");
            Require(fixture.Held is not null && fixture.Last is not null && !ReferenceEquals(fixture.Held,fixture.Last),"held output");
            VerifyOutput(fixture,fixture.Held!);
            Fill(fixture.Parent,91);Fill(fixture.Last!,-7);
            Require(Hash(fixture.Held!)==fixture.HeldHash,"independent output");
            rows.Add(new {index=fixture.Index,name=fixture.Name,shape=fixture.Shape,kind=fixture.Kind,dtype=fixture.Dtype,
                weight=fixture.Weight,batch=fixture.Batch,setup_ticks=fixture.SetupTicks,input_sha256=fixture.InputHash,
                output_sha256=fixture.HeldHash,maximum_error=fixture.MaximumError,inputs=true,ownership=true,clocks=fixture.Clocks});
        }
        long batchSum=fixtures.Sum(f=>(long)f.Batch);
        using var output=new FileStream(args[3],FileMode.CreateNew);
        JsonSerializer.Serialize(output,new {passed=true,protocol="parakeet-sigmoid-public-rounds-600-180-v1",role,sequence,
            runtime=Environment.Version.ToString(),pid=Environment.ProcessId,flags,core_sha256=core,
            assembly=FileHash(Assembly.GetExecutingAssembly().Location),census_sha256=FileHash(Path.Combine(folder,"census.json")),
            frequency=Stopwatch.Frequency,samples=780*46,measured_samples=180*46,warmup_samples=600*46,
            calls=780*batchSum,warmups=600*batchSum,measured=180*batchSum,rows},new JsonSerializerOptions{WriteIndented=true});
    }
}
