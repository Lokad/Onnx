using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text.Json;

if (args.Length != 2) throw new ArgumentException("Expected archived binary directory and new output directory.");
if (!OperatingSystem.IsWindows()) throw new PlatformNotSupportedException("This capture records Windows process affinity.");
string source = Path.GetFullPath(args[0]), output = Path.GetFullPath(args[1]);
if (Directory.Exists(output)) throw new IOException("Output already exists.");
Directory.CreateDirectory(output);
static string Sha(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
var context = new AssemblyLoadContext("ArchivedCoefficientCapture", isCollectible: true);
context.Resolving += (_, name) =>
{
    var path = Path.Combine(source, name.Name + ".dll");
    return File.Exists(path) ? context.LoadFromAssemblyPath(path) : null;
};
var assembly = context.LoadFromAssemblyPath(Path.Combine(source, "Lokad.Onnx.Data.dll"));
var type = assembly.GetType("Lokad.Onnx.WeSpeakerAudio", throwOnError: true) ?? throw new InvalidOperationException();
var fields = new List<object>();
foreach (var (name, count) in new[] { ("Window", 400), ("MelWeights", 20480) })
{
    var field = type.GetField(name, BindingFlags.NonPublic | BindingFlags.Static) ?? throw new InvalidOperationException(name);
    if (!field.IsInitOnly || field.FieldType != typeof(float[])) throw new InvalidOperationException("Unexpected field contract.");
    var values = (float[])(field.GetValue(null) ?? throw new InvalidOperationException(name));
    if (values.Length != count || !values.All(float.IsFinite)) throw new InvalidOperationException("Unexpected coefficient array.");
    var path = Path.Combine(output, name + ".f32");
    using (var stream = new FileStream(path, FileMode.CreateNew)) stream.Write(MemoryMarshal.AsBytes(values.AsSpan()));
    fields.Add(new { name, count, file = Path.GetFileName(path), sha256 = Sha(path), read_only = field.IsInitOnly });
}
var process = Process.GetCurrentProcess();
bool native = process.Modules.Cast<ProcessModule>().Any(m => m.FileName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase));
if (native) throw new InvalidOperationException("Unexpected native ORT module.");
var result = new { complete = true, source, data = new { file = assembly.Location, sha256 = Sha(assembly.Location) }, fields,
    runtime = new { pid = process.Id, framework = Environment.Version.ToString(), affinity = (long)process.ProcessorAffinity,
        processor_count = Environment.ProcessorCount, native_ort_loaded = native },
    loaded = context.Assemblies.Select(a => new { name = a.GetName().Name, file = a.Location, sha256 = Sha(a.Location) }).ToArray() };
using (var stream = new FileStream(Path.Combine(output, "result.json"), FileMode.CreateNew))
    JsonSerializer.Serialize(stream, result, new JsonSerializerOptions { WriteIndented = true });
Console.WriteLine("COEFFICIENT-CAPTURE-PASS window=400 mel=20480");
context.Unload();
