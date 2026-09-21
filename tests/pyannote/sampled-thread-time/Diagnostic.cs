using System.Diagnostics;
using System.Diagnostics.Tracing;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;
using Lokad.Onnx;

[EventSource(Name = "Lokad-Pyannote-Diagnostic")]
sealed class DiagnosticEvents : EventSource
{
    public static readonly DiagnosticEvents Log = new();
    [Event(1, Level = EventLevel.Informational)]
    public void Boundary(string phase, string name, int pass) => WriteEvent(1, phase, name, pass);
}

static class DiagnosticControl
{
    [DllImport("kernel32.dll")] internal static extern uint GetCurrentThreadId();

    public static void Wait(string output, bool sampled)
    {
        using var process = Process.GetCurrentProcess();
        var flags = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k => k.StartsWith("LOKAD_") || k.StartsWith("DOTNET_") || k.StartsWith("COMPlus_")).ToDictionary(k => k, Environment.GetEnvironmentVariable);
        Write("ready.json", new { pid = process.Id, birth_milliseconds = new DateTimeOffset(process.StartTime.ToUniversalTime()).ToUnixTimeMilliseconds(),
            affinity = process.ProcessorAffinity.ToInt64(), runtime = Environment.Version.ToString(), flags,
            warmup_records = Directory.GetFiles(output, "???.json").Length, thread_id = GetCurrentThreadId() });
        long waiting = Stopwatch.GetTimestamp();
        while (sampled && !DiagnosticEvents.Log.IsEnabled())
        {
            if (Stopwatch.GetElapsedTime(waiting).TotalSeconds > 120) throw new TimeoutException("Collector attach");
            Thread.Sleep(20);
        }
        if (sampled) Write("collector-enabled.json", new { pid = process.Id, enabled = true, ticks = Stopwatch.GetTimestamp() });
        while (!File.Exists(Path.Combine(output, "release.json")))
        {
            if (Stopwatch.GetElapsedTime(waiting).TotalSeconds > 120) throw new TimeoutException("Release");
            Thread.Sleep(20);
        }
        using var release = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(output, "release.json")));
        if (release.RootElement.GetProperty("pid").GetInt32() != process.Id || release.RootElement.GetProperty("sampled").GetBoolean() != sampled)
            throw new InvalidDataException("Release identity");
        void Write(string name, object value)
        {
            string path = Path.Combine(output, name), temporary = path + ".tmp";
            File.WriteAllText(temporary, JsonSerializer.Serialize(value, new JsonSerializerOptions { WriteIndented = true }));
            File.Move(temporary, path);
        }
    }
}

static class SampledRequests
{
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static Community1Diarization Warmup(Community1Diarizer d, float[] pcm) => d.Diarize(pcm, 16000, CancellationToken.None);

    [MethodImpl(MethodImplOptions.NoInlining)]
    public static Community1Diarization Full(Community1Diarizer d, float[] pcm)
    { var value = d.Diarize(pcm, 16000, CancellationToken.None); After(value); return value; }

    [MethodImpl(MethodImplOptions.NoInlining)]
    public static Community1Diarization FirstCrop(Community1Diarizer d, float[] pcm)
    { var value = d.Diarize(pcm, 16000, CancellationToken.None); After(value); return value; }

    [MethodImpl(MethodImplOptions.NoInlining)]
    public static Community1Diarization SecondCrop(Community1Diarizer d, float[] pcm)
    { var value = d.Diarize(pcm, 16000, CancellationToken.None); After(value); return value; }

    [MethodImpl(MethodImplOptions.NoInlining)]
    public static Community1Diarization ThirdCrop(Community1Diarizer d, float[] pcm)
    { var value = d.Diarize(pcm, 16000, CancellationToken.None); After(value); return value; }

    // Keep each scope on the sampled stack until Diarize returns; forbid a tail call.
    [MethodImpl(MethodImplOptions.NoInlining)] static void After(object value) => GC.KeepAlive(value);
}
