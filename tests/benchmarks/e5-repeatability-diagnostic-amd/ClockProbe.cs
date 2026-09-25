using System.Diagnostics;
using System.Diagnostics.Tracing;
using System.Runtime.InteropServices;
using System.Text.Json;

internal static class ClockProbe
{
    private static readonly Process Process = Process.GetCurrentProcess();
    private static readonly List<Call> Calls = new(780);
    private static Call Pending;
    private static int ThreadId;

    internal static void Initialize(string output, string key, string mode)
    {
        if ((key != "e5-8tok" && key != "e5-512tok") || mode != "timing") throw new InvalidDataException("e5 diagnostic only");
        ThreadId = gettid();
        var events = MatrixEvents.Log;
        string parent = Path.GetDirectoryName(output)!;
        File.WriteAllText(Path.Combine(parent, "ready.json"), JsonSerializer.Serialize(new Ready(Environment.ProcessId, ThreadId, Stopwatch.GetTimestamp())));
        var waiting = Stopwatch.StartNew();
        while (!events.IsEnabled(EventLevel.Informational, (EventKeywords)1))
        {
            if (waiting.Elapsed.TotalSeconds >= 30) throw new InvalidDataException("Collector did not enable markers");
            Thread.Sleep(10);
        }
        File.WriteAllText(Path.Combine(parent, "collector-enabled.json"), JsonSerializer.Serialize(new Enabled(Environment.ProcessId, Stopwatch.GetTimestamp())));
    }

    internal static void Begin(int index)
    {
        Pending = new Call { index = index, gc0 = GC.CollectionCount(0), gc1 = GC.CollectionCount(1), gc2 = GC.CollectionCount(2),
            allocated = GC.GetTotalAllocatedBytes(false), cpuBefore = Process.TotalProcessorTime.Ticks };
        Pending.marker = Stopwatch.GetTimestamp();
        MatrixEvents.Log.Begin(0, index, Pending.marker);
    }

    internal static void End(int index, long start, long end)
    {
        MatrixEvents.Log.End(0, index, end);
        Pending.cpuAfter = Process.TotalProcessorTime.Ticks;
        Pending.allocatedAfter = GC.GetTotalAllocatedBytes(false);
        Pending.after0 = GC.CollectionCount(0); Pending.after1 = GC.CollectionCount(1); Pending.after2 = GC.CollectionCount(2);
        if (Pending.index != index || index != Calls.Count) throw new InvalidDataException("Marker sequence");
        Pending.start = start; Pending.end = end;
        Calls.Add(Pending);
    }

    internal static void Save(string output)
    {
        if (Calls.Count != 780) throw new InvalidDataException("Diagnostic extent");
        File.WriteAllText(Path.Combine(output, "diagnostic.json"), JsonSerializer.Serialize(
            new Observation(true, Environment.ProcessId, ThreadId, Calls), new JsonSerializerOptions { WriteIndented = true, IncludeFields = true }));
    }

    [DllImport("libc")]
    private static extern int gettid();
    private record Ready(int pid, int native_thread, long counter);
    private record Enabled(int pid, long counter);
    private record Observation(bool diagnosticOnly, int pid, int nativeThread, List<Call> clocks);
    private struct Call
    {
        public int index, gc0, gc1, gc2, after0, after1, after2;
        public long marker, start, end, allocated, allocatedAfter, cpuBefore, cpuAfter;
    }
}

[EventSource(Name = "Lokad-Parakeet-MatMul-Diagnostic")]
internal sealed class MatrixEvents : EventSource
{
    public static readonly MatrixEvents Log = new();
    public static class Keywords { public const EventKeywords Calls = (EventKeywords)1; }
    [Event(1, Level = EventLevel.Informational, Keywords = Keywords.Calls)]
    public void Begin(int fixture, int iteration, long counter) => Emit(1, fixture, iteration, counter);
    [Event(2, Level = EventLevel.Informational, Keywords = Keywords.Calls)]
    public void End(int fixture, int iteration, long counter) => Emit(2, fixture, iteration, counter);
    [NonEvent]
    private unsafe void Emit(int id, int fixture, int iteration, long counter)
    {
        if (!IsEnabled(EventLevel.Informational, (EventKeywords)1)) return;
        EventData* data = stackalloc EventData[3];
        data[0] = new EventData { DataPointer = (IntPtr)(&fixture), Size = sizeof(int) };
        data[1] = new EventData { DataPointer = (IntPtr)(&iteration), Size = sizeof(int) };
        data[2] = new EventData { DataPointer = (IntPtr)(&counter), Size = sizeof(long) };
        WriteEventCore(id, 3, data);
    }
}
