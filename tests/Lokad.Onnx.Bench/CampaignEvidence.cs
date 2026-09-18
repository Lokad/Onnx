namespace Lokad.Onnx.Bench;

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using global::Onnx;

// All capture/hashing happens outside measured Execute/Run regions. This class
// also compiles into the normal Bench, but is inactive unless explicitly used.
internal sealed class CampaignEvidence
{
    internal static CampaignEvidence? Current { get; private set; }
    readonly string output;
    readonly string source;
    readonly string expectedCore;
    readonly string? scope;
    readonly Dictionary<string, object> cases = new(StringComparer.Ordinal);
    readonly System.Collections.Generic.List<string> failedCases = new();
    readonly Dictionary<string, (string Hash, SortedDictionary<string, string> External)> models = new(StringComparer.Ordinal);
    readonly Dictionary<string, (long Bytes, DateTime Modified)> files = new(StringComparer.Ordinal);
    Dictionary<string, object>? environment;
    long affinity;
    // Observed wall-clock runner entry (Main). This must NOT be Process.StartTime:
    // on Linux the kernel start tick truncates to the HZ boundary (up to ~10 ms early),
    // so a birth-tick timestamp can precede the supervisor's pre-spawn wall mark and
    // falsely abort the supervised interval. Wall entry is strictly inside it by construction.
    readonly string startedUtc;

    CampaignEvidence(string output, string source, string expectedCore, string? scope)
    {
        startedUtc = DateTime.UtcNow.ToString("O");
        this.output = Path.GetFullPath(output);
        this.source = source.ToLowerInvariant();
        this.expectedCore = expectedCore.ToLowerInvariant();
        if (scope is not null and not "full" and not "e5") throw new ArgumentException("Unknown evidence scope.");
        this.scope = scope;
        if (File.Exists(this.output)) throw new IOException("Evidence output already exists: " + output);
        if (!Regex.IsMatch(source, "^[0-9a-fA-F]{40}$") || !Regex.IsMatch(expectedCore, "^[0-9a-fA-F]{64}$"))
            throw new ArgumentException("Evidence requires full source commit and expected core SHA-256.");
        string actual = HashFile(typeof(ComputationalGraph).Assembly.Location);
        if (actual != this.expectedCore) throw new InvalidOperationException("Loaded core hash differs from staged core.");
    }

    internal static string[] Configure(string[] args)
    {
        var remaining = new List<string>();
        var options = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] is "--evidence-out" or "--source-sha" or "--expected-core-sha256" or "--evidence-scope")
            {
                if (i + 1 >= args.Length || !options.TryAdd(args[i], args[++i]))
                    throw new ArgumentException("Missing or duplicate evidence option.");
            }
            else remaining.Add(args[i]);
        }
        if (options.Count != 0)
        {
            if (!options.ContainsKey("--evidence-out") || !options.ContainsKey("--source-sha") || !options.ContainsKey("--expected-core-sha256"))
                throw new ArgumentException("Supply --evidence-out, --source-sha and --expected-core-sha256 together.");
            options.TryGetValue("--evidence-scope", out var scope);
            Current = new CampaignEvidence(options["--evidence-out"], options["--source-sha"], options["--expected-core-sha256"], scope);
        }
        return remaining.ToArray();
    }

    internal static string CpuIdentity()
    {
        if (OperatingSystem.IsLinux())
        {
            var line = File.ReadLines("/proc/cpuinfo").FirstOrDefault(x => x.StartsWith("model name", StringComparison.Ordinal));
            if (line != null) return line.Split(':', 2)[1].Trim();
        }
        return Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER") ?? RuntimeInformation.ProcessArchitecture.ToString();
    }

    internal void CaptureHost(long mask, string cpu)
    {
        affinity = mask;
        var variables = new SortedDictionary<string, string>(StringComparer.Ordinal);
        foreach (DictionaryEntry entry in Environment.GetEnvironmentVariables())
        {
            string key = (string)entry.Key;
            if (key.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || key.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)
                || key.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase)) variables.Add(key, (string)entry.Value!);
        }
        string? tiered = Environment.GetEnvironmentVariable("DOTNET_TieredCompilation") ?? Environment.GetEnvironmentVariable("COMPlus_TieredCompilation");
        var sdk = Assembly.GetExecutingAssembly().GetCustomAttributes<AssemblyMetadataAttribute>().FirstOrDefault(x => x.Key == "CampaignSdk")?.Value;
        if (string.IsNullOrEmpty(sdk)) throw new InvalidOperationException("Evidence requires the common Campaign project with embedded SDK identity.");
        environment = new Dictionary<string, object>
        {
            ["host"] = Environment.MachineName, ["cpu"] = cpu, ["os"] = RuntimeInformation.OSDescription,
            ["architecture"] = RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant(), ["sdk"] = sdk,
            ["runtime"] = RuntimeInformation.FrameworkDescription, ["isa"] = HardwareIntrinsics.GetFullInfo(),
            ["affinity"] = "0x" + mask.ToString("X"),
            ["settings"] = new { jit = tiered == "0" ? "full-opts" : "default-tiered",
                gc = (GCSettings.IsServerGC ? "server" : "workstation") + ":" + GCSettings.LatencyMode, variables }
        };
    }

    internal static string HashFile(string file)
    {
        using var stream = File.OpenRead(file);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    string ObserveFile(string path)
    {
        var before = new FileInfo(path);
        var stamp = (before.Length, before.LastWriteTimeUtc);
        string hash = HashFile(path);
        before.Refresh();
        if (stamp != (before.Length, before.LastWriteTimeUtc)) throw new IOException("File changed during hashing: " + path);
        files[path] = stamp;
        return hash;
    }

    internal static string HashInputs(IReadOnlyDictionary<string, ITensor> named)
    {
        using var sha = SHA256.Create();
        using var stream = new CryptoStream(Stream.Null, sha, CryptoStreamMode.Write);
        using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(Encoding.ASCII.GetBytes("LOKAD-CAMPAIGN-INPUTS-1\0"));
            writer.Write(named.Count);
            foreach (string name in named.Keys.OrderBy(x => x, StringComparer.Ordinal))
            {
                var tensor = named[name];
                byte[] bytes = Encoding.UTF8.GetBytes(name);
                writer.Write(bytes.Length); writer.Write(bytes);
                writer.Write(tensor is Tensor<long> ? 7 : tensor is Tensor<float> ? 1 : throw new NotSupportedException("Campaign input dtype"));
                writer.Write(tensor.Dims.Length);
                foreach (int dimension in tensor.Dims) writer.Write(dimension);
                writer.Write((long)tensor.Length);
                if (tensor is Tensor<long> longs) for (int i = 0; i < longs.Length; i++) writer.Write(longs.GetValue(i));
                else if (tensor is Tensor<float> floats) for (int i = 0; i < floats.Length; i++) writer.Write(floats.GetValue(i));
            }
        }
        stream.FlushFinalBlock();
        return Convert.ToHexString(sha.Hash!).ToLowerInvariant();
    }

    internal static bool SteadyWindow(IReadOnlyList<double> values)
    {
        if (values.Count < 9) return false;
        var window = values.Skip(values.Count - 9).OrderBy(x => x).ToArray();
        return window.All(x => double.IsFinite(x) && x > 0) && window[^1] - window[0] <= 0.10 * window[4];
    }

    internal static IEnumerable<TensorProto> GraphTensors(GraphProto? graph)
    {
        if (graph == null) yield break;
        foreach (var tensor in graph.Initializer) yield return tensor;
        foreach (var sparse in graph.SparseInitializer) { yield return sparse.Values; yield return sparse.Indices; }
        foreach (var node in graph.Node)
            foreach (var tensor in NodeTensors(node)) yield return tensor;
    }

    static IEnumerable<TensorProto> NodeTensors(NodeProto node)
    {
        foreach (var attr in node.Attribute)
        {
            if (attr.T != null) yield return attr.T;
            foreach (var tensor in attr.Tensors) yield return tensor;
            if (attr.SparseTensor != null) { yield return attr.SparseTensor.Values; yield return attr.SparseTensor.Indices; }
            foreach (var sparse in attr.SparseTensors) { yield return sparse.Values; yield return sparse.Indices; }
            foreach (var tensor in GraphTensors(attr.G)) yield return tensor;
            foreach (var nested in attr.Graphs) foreach (var tensor in GraphTensors(nested)) yield return tensor;
        }
    }

    internal void CaptureCase(string name, string model, Dictionary<string, ITensor> named)
    {
        model = Path.GetFullPath(model);
        if (!models.TryGetValue(model, out var identity))
        {
            string modelHash = ObserveFile(model);
            var external = new SortedDictionary<string, string>(StringComparer.Ordinal);
            using var stream = File.OpenRead(model);
            var proto = ModelProto.Parser.ParseFrom(stream);
            var tensors = GraphTensors(proto.Graph).Concat(proto.Functions.SelectMany(f => f.Node).SelectMany(NodeTensors));
            foreach (var tensor in tensors.Where(t => t.DataLocation == TensorProto.Types.DataLocation.External))
            {
                string location = tensor.ExternalData.Single(e => e.Key == "location").Value;
                string relative = location.Replace('\\', '/');
                if (Path.IsPathRooted(location) || relative.Split('/').Contains("..")) throw new IOException("External tensor must be inside the model directory.");
                if (!external.ContainsKey(relative)) external.Add(relative, ObserveFile(Path.GetFullPath(Path.Combine(Path.GetDirectoryName(model)!, location))));
            }
            identity = (modelHash, external);
            models.Add(model, identity);
        }
        var entry = new Dictionary<string, object>
        {
            ["model_sha256"] = identity.Hash, ["input_sha256"] = HashInputs(named), ["external_data"] = identity.External
        };
        if (name.StartsWith("e5-", StringComparison.Ordinal))
        {
            var mask = (Tensor<long>)named["attention_mask"];
            int count = 0;
            for (int i = 0; i < mask.Length; i++) if (mask.GetValue(i) != 0) count++;
            entry.Add("unmasked_tokens", count);
        }
        cases.Add(name, entry);
    }

    internal void SetFailedCases(System.Collections.Generic.List<string> labels)
    {
        failedCases.AddRange(labels);
    }

    internal void Finish(int exitCode)
    {
        if (environment == null) throw new InvalidOperationException("Host evidence was not captured.");
        using var process = Process.GetCurrentProcess();
        process.Refresh();
        if ((OperatingSystem.IsWindows() || OperatingSystem.IsLinux()) && process.ProcessorAffinity.ToInt64() != affinity)
            throw new InvalidOperationException("Process affinity changed during the run.");
        foreach (var pair in files)
        {
            var info = new FileInfo(pair.Key);
            if ((info.Length, info.LastWriteTimeUtc) != pair.Value) throw new IOException("Model changed during run: " + pair.Key);
        }
        var modules = process.Modules.Cast<ProcessModule>().Select(m => m.FileName)
            .Where(p => Regex.IsMatch(Path.GetFileName(p), @"^(onnxruntime\.dll|libonnxruntime\.so(\.[0-9.]+)?)$", RegexOptions.IgnoreCase))
            .Distinct(StringComparer.Ordinal).ToArray();
        if (modules.Length != 1) throw new InvalidOperationException("Expected exactly one actually loaded ORT module; found " + modules.Length);
        string corePath = typeof(ComputationalGraph).Assembly.Location;
        if (HashFile(corePath) != expectedCore) throw new IOException("Loaded core file changed during the run.");
        string runnerPath = Assembly.GetExecutingAssembly().Location;
        var runnerFiles = new SortedDictionary<string, string>(StringComparer.Ordinal);
        foreach (string path in Directory.GetFiles(Path.GetDirectoryName(runnerPath)!))
        {
            if (Path.GetFileName(path).Equals("Lokad.Onnx.dll", StringComparison.OrdinalIgnoreCase)) continue;
            if (path.EndsWith(".dll", StringComparison.OrdinalIgnoreCase) || path.EndsWith(".deps.json", StringComparison.Ordinal)
                || path.EndsWith(".runtimeconfig.json", StringComparison.Ordinal)) runnerFiles.Add(Path.GetFileName(path), HashFile(path));
        }
        // Composite identifies the common managed runner, dependencies and runtime config.
        string runnerText = string.Concat(runnerFiles.Select(p => p.Key + "\0" + p.Value + "\n"));
        string runnerHash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(runnerText))).ToLowerInvariant();
        var record = new
        {
            producer = scope is null ? "common-runner-v1" : "common-runner-v2", scope, process_id = Environment.ProcessId,
            started_utc = startedUtc, completed_utc = DateTime.UtcNow.ToString("O"),
            exit_code = exitCode, source_sha = source, core_sha256 = expectedCore, core_path = corePath,
            runner_sha256 = runnerHash, runner_files = runnerFiles,
            ort_native = new { path = modules[0], sha256 = HashFile(modules[0]), architecture = RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant() },
            environment, cases, cases_failed = failedCases
        };
        using var outputStream = new FileStream(output, FileMode.CreateNew, FileAccess.Write);
        JsonSerializer.Serialize(outputStream, record, new JsonSerializerOptions { WriteIndented = true,
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull });
    }
}
