using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.Diagnostics.Tracing;

if (args.Length != 2) throw new ArgumentException("trace-file output-directory");
string input = Path.GetFullPath(args[0]), output = Path.GetFullPath(args[1]);
if (Directory.Exists(output)) throw new IOException("Existing output");
Directory.CreateDirectory(output);
using var stream = new StreamWriter(new FileStream(Path.Combine(output, "events.jsonl"), FileMode.CreateNew));
using var source = new EventPipeEventSource(input);
var counts = new Dictionary<string, int>();
int recorded = 0, total = 0;
void Record(TraceEvent e)
{
    var payload = new Dictionary<string, string?>();
    foreach (string name in e.PayloadNames)
    {
        object? value = e.PayloadByName(name);
        payload[name] = value is IFormattable formattable
            ? formattable.ToString(null, CultureInfo.InvariantCulture) : value?.ToString();
    }
    stream.WriteLine(JsonSerializer.Serialize(new { index = recorded++, provider = e.ProviderName,
        name = e.EventName, id = (int)e.ID, version = e.Version, thread = e.ThreadID, pid = e.ProcessID,
        ms = e.TimeStampRelativeMSec, payload }));
}
source.Clr.All += e =>
{
    total++;
    counts[e.EventName] = counts.GetValueOrDefault(e.EventName) + 1;
    if (e.EventName.StartsWith("GC/", StringComparison.Ordinal)) Record(e);
};
source.Dynamic.All += e =>
{
    if (e.ProviderName == "Lokad-Pyannote-Diagnostic") Record(e);
};
source.Process();
stream.Flush();
using var trace = File.OpenRead(input);
string hash = Convert.ToHexStringLower(SHA256.HashData(trace));
File.WriteAllText(Path.Combine(output, "summary.json"), JsonSerializer.Serialize(new {
    complete = true, input_sha256 = hash, runtime = Environment.Version.ToString(), exporter_pid = Environment.ProcessId,
    lost = source.EventsLost, clr_events = total, recorded, counts
}, new JsonSerializerOptions { WriteIndented = true }));
if (source.EventsLost != 0 || total == 0 || recorded == 0) throw new InvalidDataException("Incomplete event stream");
