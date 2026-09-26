static class MemoryCounters
{
    // Linux x86-64 rusage: two timeval pairs followed by fourteen long fields.
    [StructLayout(LayoutKind.Sequential)]
    public struct Usage
    {
        public long UserSeconds, UserMicros, SystemSeconds, SystemMicros;
        public long MaxRss, Shared, Unshared, Stack, Minor, Major, Swaps;
        public long InputBlocks, OutputBlocks, Sent, Received, Signals, Voluntary, Involuntary;
    }

    public readonly record struct Snapshot(long Begin, Usage Usage,
        long Allocated, int Gc0, int Gc1, int Gc2);

    [DllImport("libc", SetLastError = true)]
    static extern int getrusage(int who, out Usage usage);
    [DllImport("libc")]
    public static extern int gettid();

    public static Snapshot Before()
    {
        long began = Stopwatch.GetTimestamp();
        if (getrusage(1, out var usage) != 0) throw new InvalidOperationException("RUSAGE_THREAD");
        int gc0 = GC.CollectionCount(0), gc1 = GC.CollectionCount(1), gc2 = GC.CollectionCount(2);
        long allocated = GC.GetAllocatedBytesForCurrentThread();
        return new(began, usage, allocated, gc0, gc1, gc2);
    }

    public static object After(Snapshot before, long start, long stop)
    {
        long allocated = GC.GetAllocatedBytesForCurrentThread();
        int gc0 = GC.CollectionCount(0), gc1 = GC.CollectionCount(1), gc2 = GC.CollectionCount(2);
        if (getrusage(1, out var after) != 0) throw new InvalidOperationException("RUSAGE_THREAD");
        long ended = Stopwatch.GetTimestamp();
        return new
        {
            begin = before.Begin, start, stop, end = ended,
            allocatedBefore = before.Allocated, allocatedAfter = allocated,
            gc0 = before.Gc0, gc1 = before.Gc1, gc2 = before.Gc2,
            after0 = gc0, after1 = gc1, after2 = gc2,
            userBeforeUs = before.Usage.UserSeconds * 1000000 + before.Usage.UserMicros,
            userAfterUs = after.UserSeconds * 1000000 + after.UserMicros,
            systemBeforeUs = before.Usage.SystemSeconds * 1000000 + before.Usage.SystemMicros,
            systemAfterUs = after.SystemSeconds * 1000000 + after.SystemMicros,
            minorBefore = before.Usage.Minor, minorAfter = after.Minor,
            majorBefore = before.Usage.Major, majorAfter = after.Major,
            voluntaryBefore = before.Usage.Voluntary, voluntaryAfter = after.Voluntary,
            involuntaryBefore = before.Usage.Involuntary, involuntaryAfter = after.Involuntary
        };
    }

    public static void Calibrate(string outputDirectory, string phase)
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ||
            RuntimeInformation.OSArchitecture != Architecture.X64 || Marshal.SizeOf<Usage>() != 144)
            throw new InvalidOperationException("Linux x86-64 rusage ABI");
        var clocks = new List<object>(128);
        for (int iteration = 0; iteration < 128; iteration++)
        {
            var before = Before();
            long start = Stopwatch.GetTimestamp();
            long stop = Stopwatch.GetTimestamp();
            clocks.Add(new { iteration, ticks = stop - start, counters = After(before, start, stop) });
        }
        using var output = new FileStream(Path.Combine(outputDirectory, "calibration-" + phase + ".json"), FileMode.CreateNew);
        JsonSerializer.Serialize(output, new { diagnosticOnly = true, phase, pid = Environment.ProcessId,
            nativeThread = gettid(), frequency = Stopwatch.Frequency, rusageBytes = Marshal.SizeOf<Usage>(), clocks },
            new JsonSerializerOptions { WriteIndented = true });
    }
}
