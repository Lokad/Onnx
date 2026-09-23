using System.Diagnostics;
using System.Diagnostics.Tracing;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

static class Screen
{
    [DllImport("libc")] static extern int gettid();
    static void Require(bool value, string message) { if (!value) throw new InvalidOperationException(message); }
    static string Hash(ReadOnlySpan<float> data) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(data)));
    static string FileHash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
    static int[] Ints(JsonElement element) => element.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    // Independent destination-coordinate oracle, including the retained crop
    // and reflection controls. No row-copy helper or product tensor indexing.
    static float[] Oracle(float[] input, int[] shape, int[] pads, float fill, bool reflect)
    {
        int rank = shape.Length;
        int[] dimensions = shape.Select((v, axis) => v + pads[axis] + pads[rank + axis]).ToArray();
        var result = new float[dimensions.Aggregate(1, (a, b) => checked(a * b))];
        for (int flat = 0; flat < result.Length; flat++)
        {
            int rest = flat, offset = 0, stride = 1;
            bool inside = true;
            for (int axis = rank - 1; axis >= 0; axis--)
            {
                int coordinate = rest % dimensions[axis]; rest /= dimensions[axis];
                int index;
                if (reflect)
                {
                    int start = Math.Max(-pads[axis], 0);
                    int retained = shape[axis] + Math.Min(pads[axis], 0) + Math.Min(pads[rank + axis], 0);
                    int relative = coordinate - Math.Max(pads[axis], 0);
                    int period = 2 * (retained - 1);
                    int folded = retained == 1 ? 0 : (relative % period + period) % period;
                    index = start + (folded < retained ? folded : period - folded);
                }
                else
                {
                    index = coordinate - pads[axis];
                    inside &= index >= 0 && index < shape[axis];
                }
                offset += index * stride; stride *= shape[axis];
            }
            result[flat] = inside ? input[offset] : fill;
        }
        return result;
    }

    static void Main(string[] args)
    {
        Require(args.Length == 4, "base role sequence output");
        string folder = Path.GetFullPath(args[0]), role = args[1]; int sequence = int.Parse(args[2]);
        Require(new[] { "current", "candidate" }[sequence] == role, "order");
        using var process = Process.GetCurrentProcess();
        Require(Environment.Version.ToString() == "10.0.8" && Environment.ProcessorCount == 1 && process.ProcessorAffinity == (nint)4, "runtime/affinity");
        Require(Avx2.IsSupported && Fma.IsSupported && Avx512F.IsSupported, "ISA");
        var flags = Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>()
            .Where(e => new[] { "LOKAD_", "DOTNET_", "COMPlus_" }.Any(p => ((string)e.Key).StartsWith(p, StringComparison.OrdinalIgnoreCase)))
            .ToDictionary(e => (string)e.Key, e => (string)e.Value!);
        Require(flags.Count == 0, "ordinary runtime flags");
        using var payloadDocument = JsonDocument.Parse(File.ReadAllText(Path.Combine(folder, "payload.json")));
        using var censusDocument = JsonDocument.Parse(File.ReadAllText(Path.Combine(folder, "census.json")));
        var payload = payloadDocument.RootElement;
        string core = FileHash(typeof(Tensor<>).Assembly.Location);
        Require(core == payload.GetProperty("products").GetProperty(role).GetProperty("Lokad.Onnx.dll").GetProperty("sha256").GetString(), "actual product");
        string outputDirectory = Path.GetDirectoryName(Path.GetFullPath(args[3]))!;
        var events = PadEvents.Log; int workerThread = gettid();
        File.WriteAllText(Path.Combine(outputDirectory, "ready.json"), JsonSerializer.Serialize(new
            { pid = Environment.ProcessId, native_thread = workerThread, counter = Stopwatch.GetTimestamp() }));
        long waitStart = Stopwatch.GetTimestamp();
        while (!events.IsEnabled(EventLevel.Informational, (EventKeywords)1))
        {
            Require(Stopwatch.GetElapsedTime(waitStart).TotalSeconds < 30, "collector startup timeout");
            Thread.Sleep(10);
        }
        File.WriteAllText(Path.Combine(outputDirectory, "collector-enabled.json"), JsonSerializer.Serialize(new
            { pid = Environment.ProcessId, counter = Stopwatch.GetTimestamp() }));
        var rows = new List<object>(); int index = 0;
        foreach (var entry in censusDocument.RootElement.GetProperty("cases").EnumerateArray())
        {
            long setup = Stopwatch.GetTimestamp();
            int[] shape = Ints(entry.GetProperty("shape")), pads = Ints(entry.GetProperty("pads"));
            int[] originalPads = (int[])pads.Clone();
            float fill = entry.GetProperty("fill").GetSingle();
            string mode = entry.GetProperty("mode").GetString()!;
            var input = Enumerable.Range(0, shape.Aggregate(1, (a, b) => checked(a * b))).Select(i => (i % 257 - 128) / 8f).ToArray();
            var expected = Oracle(input, shape, pads, fill, mode == "reflect");
            string inputHash = Hash(input), expectedHash = Hash(expected);
            int[] dimensions = shape.Select((v, axis) => v + pads[axis] + pads[shape.Length + axis]).ToArray();
            var source = new DenseTensor<float>(input, shape);
            var padTensor = new DenseTensor<int>(pads, new[] { pads.Length });
            var fillTensor = DenseTensor<float>.Scalar(fill);
            long setupTicks = Stopwatch.GetTimestamp() - setup;
            var clocks = new List<object>(780);
            DenseTensor<float>? held = null, last = null;
            for (int iteration = 0; iteration < 780; iteration++)
            {
                int gc0 = GC.CollectionCount(0), gc1 = GC.CollectionCount(1), gc2 = GC.CollectionCount(2);
                long allocated = GC.GetAllocatedBytesForCurrentThread();
                long totalAllocated = GC.GetTotalAllocatedBytes(false);
                long marker = Stopwatch.GetTimestamp();
                events.Begin(index, iteration, marker);
                long start = Stopwatch.GetTimestamp();
                var returned = CPUExecutionProvider.Pad(source, padTensor, fillTensor, mode, null, null, null);
                long stop = Stopwatch.GetTimestamp();
                events.End(index, iteration, stop);
                long allocatedAfter = GC.GetAllocatedBytesForCurrentThread();
                long totalAllocatedAfter = GC.GetTotalAllocatedBytes(false);
                int after0 = GC.CollectionCount(0), after1 = GC.CollectionCount(1), after2 = GC.CollectionCount(2);
                long ticks = stop - start;
                Require(ticks > 0 && returned.Status == OpStatus.Success, "clock/status");
                last = (DenseTensor<float>)returned.Outputs![0];
                clocks.Add(new { iteration, warmup = iteration < 600, marker, start, stop, ticks,
                    gc0, gc1, gc2, after0, after1, after2, allocated, allocatedAfter, totalAllocated, totalAllocatedAfter });
                if (iteration == 0) held = last;
                if (iteration is 0 or 599 or 779)
                {
                    Require(last.Dimensions.SequenceEqual(dimensions), "shape");
                    Require(MemoryMarshal.AsBytes(last.Buffer.Span).SequenceEqual(MemoryMarshal.AsBytes(expected.AsSpan())), "all output bits");
                }
            }
            Require(Hash(input) == inputHash && padTensor.ToArray().SequenceEqual(originalPads), "input ownership");
            Require(fillTensor.GetValue(0) == fill, "fill ownership");
            Require(held is not null && last is not null && !ReferenceEquals(held, last), "held output");
            input.AsSpan().Clear(); last!.Buffer.Span.Fill(9876f);
            Require(Hash(held!.Buffer.Span) == expectedHash, "independent returned storage");
            rows.Add(new { index = index++, name = entry.GetProperty("name").GetString(), shape, pads, mode, fill,
                setupTicks, output = expectedHash, exact = true, inputs = true, ownership = true, clocks });
        }
        Require(index == 12, "fixed census");
        using var output = new FileStream(args[3], FileMode.CreateNew);
        JsonSerializer.Serialize(output, new { passed = true, protocol = "parakeet-pad-runtime-diagnostic-v1", diagnosticOnly = true, role, sequence,
            runtime = Environment.Version.ToString(), pid = Environment.ProcessId, nativeThread = workerThread, flags, core_sha256 = core,
            assembly = FileHash(Assembly.GetExecutingAssembly().Location), frequency = Stopwatch.Frequency,
            calls = 9360, warmups = 7200, measured = 2160, rows }, new JsonSerializerOptions { WriteIndented = true });
    }
}

[EventSource(Name = "Lokad-Parakeet-Pad-Diagnostic")]
sealed class PadEvents : EventSource
{
    public static readonly PadEvents Log = new();
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
