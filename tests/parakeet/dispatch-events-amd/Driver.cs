using System.Diagnostics;
using System.Diagnostics.Tracing;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

static class Driver
{
    static void Require(bool value, string message) { if (!value) throw new InvalidOperationException(message); }
    static string Hash(ReadOnlySpan<float> data) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(data)));
    static string FileHash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
    static float[] Read(string folder, JsonElement descriptor)
    {
        byte[] bytes = File.ReadAllBytes(Path.Combine(folder,descriptor.GetProperty("file").GetString()!));
        Require(bytes.Length == descriptor.GetProperty("bytes").GetInt32() && Convert.ToHexStringLower(SHA256.HashData(bytes)) == descriptor.GetProperty("sha256").GetString(), "fixture bytes/identity");
        return MemoryMarshal.Cast<byte,float>(bytes).ToArray();
    }
    [DllImport("libc")]
    static extern int gettid();
    static unsafe object Addresses(float[] a, float[] b, DenseTensor<float> destination)
    {
        fixed (float* ap = a, bp = b)
        using (var cp = destination.Buffer.Pin())
            return new { a = (ulong)ap, b = (ulong)bp, c = (ulong)cp.Pointer };
    }
    static void Main(string[] args)
    {
        Require(args.Length == 4, "base role sequence output");
        string folder = Path.GetFullPath(args[0]), role = args[1]; int sequence = int.Parse(args[2]);
        Require(new[] {"current","candidate"}[sequence] == role, "process order");
        var process = Process.GetCurrentProcess();
        Require(Environment.Version.ToString() == "10.0.8" && Environment.ProcessorCount == 1 && process.ProcessorAffinity == (nint)4, "runtime/affinity");
        Require(Avx2.IsSupported && Fma.IsSupported && Avx512F.IsSupported, "ISA");
        var flags = Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>().Where(e => new[] {"LOKAD_","DOTNET_","COMPlus_"}.Any(p => ((string)e.Key).StartsWith(p,StringComparison.OrdinalIgnoreCase))).ToDictionary(e => (string)e.Key,e => (string)e.Value!);
        Require(flags.Count == 0, "ordinary runtime flags");
        var payload = JsonDocument.Parse(File.ReadAllText(Path.Combine(folder,"payload.json"))).RootElement;
        string core = FileHash(typeof(Tensor<>).Assembly.Location);
        Require(core == payload.GetProperty("products").GetProperty(role).GetProperty("Lokad.Onnx.dll").GetProperty("sha256").GetString(), "actual product");
        string outputDirectory = Path.GetDirectoryName(Path.GetFullPath(args[3]))!;
        var events = MatrixEvents.Log;
        int workerThread = gettid();
        File.WriteAllText(Path.Combine(outputDirectory,"ready.json"),JsonSerializer.Serialize(new { pid = Environment.ProcessId, native_thread = workerThread, counter = Stopwatch.GetTimestamp() }));
        var waiting = Stopwatch.StartNew();
        while (!events.IsEnabled(EventLevel.Informational,(EventKeywords)1))
        {
            Require(waiting.Elapsed.TotalSeconds < 30,"collector did not enable markers");
            Thread.Sleep(10);
        }
        File.WriteAllText(Path.Combine(outputDirectory,"collector-enabled.json"),JsonSerializer.Serialize(new { pid = Environment.ProcessId, counter = Stopwatch.GetTimestamp() }));
        var capture = JsonDocument.Parse(File.ReadAllText(Path.Combine(folder,"fixtures/result.json"))).RootElement;
        var rows = new List<object>(); int index = 0;
        foreach (var entry in capture.GetProperty("entries").EnumerateArray())
        {
            long preparation = Stopwatch.GetTimestamp();
            int m = entry.GetProperty("m").GetInt32(), n = entry.GetProperty("k").GetInt32(), k = entry.GetProperty("n").GetInt32();
            var a = Read(Path.Combine(folder,"fixtures"),entry.GetProperty("a"));
            var b = Read(Path.Combine(folder,"fixtures"),capture.GetProperty("weights").GetProperty(entry.GetProperty("weight").GetString()!));
            var expected = Read(Path.Combine(folder,"fixtures"),entry.GetProperty("y"));
            string ah = Hash(a), bh = Hash(b), expectedHash = Hash(expected);
            var x = new DenseTensor<float>(a,new[] {m,n}); var y = new DenseTensor<float>(b,new[] {n,k});
            var storage = Enumerable.Repeat(-12345.5f,m*k+6).ToArray();
            var destination = new DenseTensor<float>(storage.AsMemory(3,m*k),new[] {m,k});
            var options = TensorExecutionOptions.Intrinsics;
            var addressesBefore = Addresses(a,b,destination);
            long preparationTicks = Stopwatch.GetTimestamp()-preparation;
            var clocks = new List<object>(120);
            for (int iteration = 0; iteration < 120; iteration++)
            {
                int gc0 = GC.CollectionCount(0), gc1 = GC.CollectionCount(1), gc2 = GC.CollectionCount(2);
                long allocated = GC.GetTotalAllocatedBytes(false);
                long marker = Stopwatch.GetTimestamp();
                events.Begin(index,iteration,marker);
                long start = Stopwatch.GetTimestamp();
                var returned = Tensor<float>.MatMul2D(x,y,destination,options);
                long stop = Stopwatch.GetTimestamp();
                events.End(index,iteration,stop);
                long allocatedAfter = GC.GetTotalAllocatedBytes(false);
                int after0 = GC.CollectionCount(0), after1 = GC.CollectionCount(1), after2 = GC.CollectionCount(2);
                Require(ReferenceEquals(destination,returned) && stop > start && start >= marker,"clock/destination");
                clocks.Add(new { iteration,warmup = iteration < 60,marker,start,stop,ticks = stop-start,
                    gc0,gc1,gc2,after0,after1,after2,allocated,allocatedAfter });
                if (iteration == 59 || iteration == 119)
                {
                    Require(MemoryMarshal.AsBytes(destination.Buffer.Span).SequenceEqual(MemoryMarshal.AsBytes(expected.AsSpan())),"full output bits");
                    Require(storage.Take(3).Concat(storage.TakeLast(3)).All(v => v == -12345.5f),"guards");
                }
            }
            Require(Hash(a) == ah && Hash(b) == bh,"input mutation");
            rows.Add(new {index = index++,name = entry.GetProperty("name").GetString(),node = entry.GetProperty("node").GetInt32(),
                m,reduction = n,columns = k,preparationTicks,addressesBefore,addressesAfter = Addresses(a,b,destination),output = expectedHash,exact = true,guards = true,inputs = true,clocks});
        }
        Require(index == 21,"fixed fixture census");
        File.WriteAllText(args[3],JsonSerializer.Serialize(new {passed = true,protocol = "parakeet-dispatch-events-v1",diagnosticOnly = true,role,sequence,
            runtime = Environment.Version.ToString(),pid = Environment.ProcessId,nativeThread = workerThread,flags,core_sha256 = core,
            assembly = FileHash(Assembly.GetExecutingAssembly().Location),frequency = Stopwatch.Frequency,
            calls = 2520,warmups = 1260,measured = 1260,rows},new JsonSerializerOptions {WriteIndented = true}));
    }
}

[EventSource(Name = "Lokad-Parakeet-MatMul-Diagnostic")]
sealed class MatrixEvents : EventSource
{
    public static readonly MatrixEvents Log = new();
    public static class Keywords { public const EventKeywords Calls = (EventKeywords)1; }
    [Event(1, Level = EventLevel.Informational, Keywords = Keywords.Calls)]
    public void Begin(int fixture, int iteration, long counter) => Emit(1,fixture,iteration,counter);
    [Event(2, Level = EventLevel.Informational, Keywords = Keywords.Calls)]
    public void End(int fixture, int iteration, long counter) => Emit(2,fixture,iteration,counter);
    [NonEvent]
    unsafe void Emit(int id, int fixture, int iteration, long counter)
    {
        if (!IsEnabled(EventLevel.Informational,(EventKeywords)1)) return;
        EventData* data = stackalloc EventData[3];
        data[0] = new EventData { DataPointer = (IntPtr)(&fixture), Size = sizeof(int) };
        data[1] = new EventData { DataPointer = (IntPtr)(&iteration), Size = sizeof(int) };
        data[2] = new EventData { DataPointer = (IntPtr)(&counter), Size = sizeof(long) };
        WriteEventCore(id,3,data);
    }
}
