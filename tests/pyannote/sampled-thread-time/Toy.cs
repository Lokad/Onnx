using System.Diagnostics;
using System.Diagnostics.Tracing;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;

if (args.Length != 2) throw new ArgumentException("new-output sampled|control");
string output = Path.GetFullPath(args[0]); bool sampled = args[1] == "sampled";
if (!sampled && args[1] != "control") throw new ArgumentException("mode");
if (Directory.Exists(output)) throw new IOException("Existing output");
Directory.CreateDirectory(output);
using var process = Process.GetCurrentProcess();
if (process.ProcessorAffinity.ToInt64() != 4 || Environment.ProcessorCount != 1) throw new InvalidOperationException("CPU2");
ToyMarkers.Warmup();
var flags = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k => k.StartsWith("LOKAD_") || k.StartsWith("DOTNET_") || k.StartsWith("COMPlus_")).ToDictionary(k => k, Environment.GetEnvironmentVariable);
Write("ready.json", new { pid = process.Id, birth_milliseconds = new DateTimeOffset(process.StartTime.ToUniversalTime()).ToUnixTimeMilliseconds(),
    affinity = process.ProcessorAffinity.ToInt64(), runtime = Environment.Version.ToString(), flags });
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
using (var release = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(output, "release.json"))))
    if (release.RootElement.GetProperty("pid").GetInt32() != process.Id || release.RootElement.GetProperty("sampled").GetBoolean() != sampled)
        throw new InvalidDataException("Release identity");
var records = new List<object>();
for (int pass = 0; pass < 2; pass++)
{
    DiagnosticEvents.Log.Boundary("begin", pass); long begin = Stopwatch.GetTimestamp();
    process.Refresh(); long cpuBefore = process.TotalProcessorTime.Ticks;
    double value = ToyMarkers.Measured();
    process.Refresh(); long cpuAfter = process.TotalProcessorTime.Ticks;
    long end = Stopwatch.GetTimestamp(); DiagnosticEvents.Log.Boundary("end", pass);
    records.Add(new { pass, begin, end, seconds = (end-begin)/(double)Stopwatch.Frequency, cpu_ticks = cpuAfter-cpuBefore, value });
}
ToyMarkers.Validation();
Write("result.json", new { passed = true, sampled, records, frequency = Stopwatch.Frequency, runtime = Environment.Version.ToString(), flags });
void Write(string name, object value)
{
    string path = Path.Combine(output, name), temporary = path + ".tmp";
    File.WriteAllText(temporary, JsonSerializer.Serialize(value, new JsonSerializerOptions { WriteIndented = true }));
    File.Move(temporary, path);
}

[EventSource(Name = "Lokad-Pyannote-Diagnostic")]
sealed class DiagnosticEvents : EventSource
{
    public static readonly DiagnosticEvents Log = new();
    [Event(1, Level = EventLevel.Informational)]
    public void Boundary(string phase, int pass) => WriteEvent(1, phase, pass);
}

static class ToyMarkers
{
    [MethodImpl(MethodImplOptions.NoInlining)] public static double Warmup() => Burn(1);
    [MethodImpl(MethodImplOptions.NoInlining)] public static double Measured() => Burn(2);
    [MethodImpl(MethodImplOptions.NoInlining)] public static double Validation() => Burn(1);
    [MethodImpl(MethodImplOptions.NoInlining)] static double Burn(int seconds)
    {
        long start = Stopwatch.GetTimestamp(); double value = 1;
        while (Stopwatch.GetElapsedTime(start).TotalSeconds < seconds)
            for (int i = 0; i < 10000; i++) value = Math.Sqrt(value + 1.25);
        return value;
    }
}
