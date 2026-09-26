using System.Diagnostics;
using System.Diagnostics.Tracing;
using System.Text.Json;

[EventSource(Name = "Lokad-Parakeet-Clock")]
sealed class ParakeetClock : EventSource
{
    internal static readonly ParakeetClock Log = new();
    [Event(1, Level = EventLevel.Informational)]
    public void Anchor(long counter) => WriteEvent(1, counter);
}

static class TraceStartup
{
    static readonly List<ClockAnchor> Anchors = new();

    internal static void Wait(string output)
    {
        using var process = Process.GetCurrentProcess();
        // Instantiate both providers before announcing readiness.
        _ = DiagnosticEvents.Log.IsEnabled();
        _ = ParakeetClock.Log.IsEnabled();
        Write(output, "startup-ready.json", new { pid = process.Id,
            birth_milliseconds = new DateTimeOffset(process.StartTime.ToUniversalTime()).ToUnixTimeMilliseconds(),
            thread_id = DiagnosticControl.GetCurrentThreadId(), counter = Stopwatch.GetTimestamp() });
        long waiting = Stopwatch.GetTimestamp();
        while (!DiagnosticEvents.Log.IsEnabled() || !ParakeetClock.Log.IsEnabled())
        {
            if (Stopwatch.GetElapsedTime(waiting).TotalSeconds > 120)
                throw new TimeoutException("Collector attach before model construction");
            Thread.Sleep(20);
        }
        Write(output, "startup-enabled.json", new { pid = process.Id, counter = Stopwatch.GetTimestamp() });
        Anchor();
    }

    internal static void Anchor()
    {
        if (!ParakeetClock.Log.IsEnabled()) throw new InvalidDataException("Clock collector stopped");
        long before = Stopwatch.GetTimestamp();
        ParakeetClock.Log.Anchor(before);
        long after = Stopwatch.GetTimestamp();
        Anchors.Add(new ClockAnchor(before, after, DiagnosticControl.GetCurrentThreadId()));
    }

    internal static void Save(string output) => Write(output, "clock-anchors.json",
        new { frequency = Stopwatch.Frequency, anchors = Anchors });

    static void Write(string output, string name, object value)
    {
        string target = Path.Combine(output, name), temporary = target + ".tmp";
        using (var stream = new FileStream(temporary, FileMode.CreateNew))
            JsonSerializer.Serialize(stream, value);
        File.Move(temporary, target);
    }

    sealed record ClockAnchor(long before, long after, uint thread);
}
