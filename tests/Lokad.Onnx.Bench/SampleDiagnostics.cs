using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace Lokad.Onnx.Bench;

/// <summary>
/// Optional observations around benchmark calls, never campaign evidence.
/// Reads and record storage stay outside the existing headline stopwatch.
/// GC observations are process-wide; CPU time belongs only to this thread.
/// Neither identifies a pause cause without comparing the actual samples.
/// </summary>
internal sealed class SampleDiagnostics
{
    internal readonly record struct Snapshot(long Timestamp, long? ThreadCpuNs, long? ProcessCpuNs,
        int Gen0, int Gen1, int Gen2, long PauseTicks, long AllocatedBytes)
    {
        internal static Snapshot Capture() => new(Stopwatch.GetTimestamp(), CpuNanoseconds(true), CpuNanoseconds(false),
            GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2),
            GC.GetTotalPauseDuration().Ticks, GC.GetAllocatedBytesForCurrentThread());
    }

    readonly record struct Row(string Engine, int Index, int Order, long StartTimestamp,
        double WallMs, double ObservationMs, double? ThreadCpuMs, double? ProcessCpuMs, int Gen0, int Gen1, int Gen2,
        double GcPauseMs, long ThreadAllocatedBytes);

    readonly List<Row> rows;

    internal SampleDiagnostics(int iterations) => rows = new List<Row>(checked(iterations * 5));

    internal void Record(string engine, int index, long elapsedTicks, Snapshot before)
    {
        var after = Snapshot.Capture();
        rows.Add(new Row(engine, index, rows.Count, before.Timestamp,
            1000.0 * elapsedTicks / Stopwatch.Frequency,
            1000.0 * (after.Timestamp - before.Timestamp) / Stopwatch.Frequency,
            (after.ThreadCpuNs - before.ThreadCpuNs) / 1_000_000.0,
            (after.ProcessCpuNs - before.ProcessCpuNs) / 1_000_000.0,
            after.Gen0 - before.Gen0, after.Gen1 - before.Gen1, after.Gen2 - before.Gen2,
            (after.PauseTicks - before.PauseTicks) / (double)TimeSpan.TicksPerMillisecond,
            after.AllocatedBytes - before.AllocatedBytes));
    }

    internal void Write(string name)
    {
        Console.WriteLine("sample-diagnostics " + JsonSerializer.Serialize(new
        {
            schema = 1,
            scope = "diagnostic-only; observations surround the stopwatch; GC is process-wide; allocation is current-thread; CPU includes separate thread/process counters (coarse on Windows); no subtraction-based corrected latency",
            @case = name,
            stopwatchFrequency = Stopwatch.Frequency,
            samples = rows
        }));
    }

    [StructLayout(LayoutKind.Sequential)]
    struct Timespec { internal long Seconds; internal long Nanoseconds; }

    [DllImport("libc", EntryPoint = "clock_gettime", SetLastError = true)]
    static extern int ClockGetTime(int clock, out Timespec time);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    static extern bool GetThreadTimes(nint thread, out long creation, out long exit, out long kernel, out long user);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    static extern bool GetProcessTimes(nint process, out long creation, out long exit, out long kernel, out long user);

    static long? CpuNanoseconds(bool thread)
    {
        if (OperatingSystem.IsLinux())
        {
            const int ClockThreadCpuTimeId = 3, ClockProcessCpuTimeId = 2;
            if (ClockGetTime(thread ? ClockThreadCpuTimeId : ClockProcessCpuTimeId, out var time) != 0)
                throw new InvalidOperationException("clock_gettime failed: " + Marshal.GetLastPInvokeError());
            return checked(time.Seconds * 1_000_000_000 + time.Nanoseconds);
        }
        if (OperatingSystem.IsWindows())
        {
            // Current-thread/process pseudo handles. FILETIME units are 100 ns.
            long kernel, user;
            bool ok = thread ? GetThreadTimes(new nint(-2), out _, out _, out kernel, out user)
                : GetProcessTimes(new nint(-1), out _, out _, out kernel, out user);
            if (!ok) throw new InvalidOperationException("CPU time read failed: " + Marshal.GetLastPInvokeError());
            return checked((kernel + user) * 100);
        }
        return null;
    }
}
