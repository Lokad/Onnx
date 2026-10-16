using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using Lokad.Onnx;
using Lokad.Onnx.Bench;
using Lokad.Tokenizers.Tokenizer;
using Microsoft.ML.OnnxRuntime;

internal static class Program
{
    internal const string Protocol = "embeddinggemma2-owned-vector-v1";
    internal const string Contract = "reused-context-or-optimized-original-ORT; owned-final-vector; pool/truncate/normalize/copy inside; reset/disposal outside; 33 calls retained";
    static readonly JsonSerializerOptions Json = new() { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow, RespectRequiredConstructorParameters = true, WriteIndented = true };
    sealed record Configuration(string Root, string Fixture, string FixtureSha256, string Case, int Dimensions,
        int Cpu, string Output, string SourceSha, string CoreSha256, long PackedWeightBudget, bool Smoke);
    static readonly Dictionary<string, (int Width, int Real)> Cases = new(StringComparer.Ordinal) {
        ["gemma2-32tok"] = (32, 32), ["gemma2-128tok"] = (128, 128), ["gemma2-512tok"] = (512, 512), ["gemma2-32pad128"] = (128, 32) };

    static int Main(string[] args)
    {
        try { Run(args); return 0; }
        catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }

    static void Require(bool value, string message)
    {
        if (!value) throw new InvalidDataException(message);
    }

    static long CurrentAffinity()
    {
        if (OperatingSystem.IsLinux() || OperatingSystem.IsWindows())
        {
            using var process = Process.GetCurrentProcess();
            return process.ProcessorAffinity.ToInt64();
        }
        throw new PlatformNotSupportedException("Affinity requires Linux or Windows");
    }

    static T Read<T>(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(path));
        void Keys(JsonElement element)
        {
            if (element.ValueKind == JsonValueKind.Object)
            {
                var names = new HashSet<string>(StringComparer.Ordinal);
                foreach (var property in element.EnumerateObject()) { Require(names.Add(property.Name), "Duplicate JSON key"); Keys(property.Value); }
            }
            else if (element.ValueKind == JsonValueKind.Array) foreach (var item in element.EnumerateArray()) Keys(item);
        }
        Keys(document.RootElement);
        return document.RootElement.Deserialize<T>(Json) ?? throw new InvalidDataException("Empty configuration");
    }

    static T[] Array<T>(string directory, JsonElement info, int width, string dtype) where T : unmanaged
    {
        string file = info.GetProperty("file").GetString()!;
        Require(Path.GetFileName(file) == file, "Fixture file must be a basename");
        string path = Path.Combine(directory, file);
        Require(CampaignEvidence.HashFile(path) == info.GetProperty("sha256").GetString(), "Fixture digest differs");
        Require(info.GetProperty("dtype").GetString() == dtype && info.GetProperty("shape").EnumerateArray().Select(d => d.GetInt32()).SequenceEqual(new[] { 1, width }), "Fixture tensor contract differs");
        byte[] bytes = File.ReadAllBytes(path);
        Require(bytes.Length == checked(width * Marshal.SizeOf<T>()) && BitConverter.IsLittleEndian, "Fixture byte count/endianness differs");
        return MemoryMarshal.Cast<byte, T>(bytes).ToArray();
    }

    static SessionOptions NativeOptions()
    {
        var settings = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            IntraOpNumThreads = 1, InterOpNumThreads = 1, ExecutionMode = ExecutionMode.ORT_SEQUENTIAL };
        settings.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
        settings.AddSessionConfigEntry("session.inter_op.allow_spinning", "0");
        return settings;
    }

    static float[] NativeVector(OrtValue value, int dimensions)
    {
        Require(value.GetTensorTypeAndShape().Shape.SequenceEqual(new long[] { 1, 768 }), "Native vector shape differs");
        float[] vector = value.GetTensorDataAsSpan<float>()[..dimensions].ToArray();
        if (dimensions != 768)
        {
            float squared = 0;
            foreach (float x in vector) squared += x * x;
            float norm = MathF.Sqrt(squared);
            for (int i = 0; i < vector.Length; i++) vector[i] /= norm;
        }
        return vector;
    }

    sealed class Engine : IDisposable
    {
        readonly GraphExecution? graph;
        readonly ComputationalGraph? prepared;
        readonly InferenceSession? session;
        readonly SessionOptions? settings;
        readonly RunOptions? run;
        readonly Dictionary<string, ITensor> inputs;
        readonly Dictionary<string, OrtValue> native = new(StringComparer.Ordinal);
        readonly Tensor<long> mask;
        readonly int dimensions;
        readonly string[] outputNames = new[] { "sentence_embedding" };
        IDisposableReadOnlyCollection<OrtValue>? result;
        internal float[]? Vector;
        internal long PackedBytes => Math.Max(prepared?.RetainedPackedWeightBytes ?? 0, graph?.RetainedPackedWeightBytes ?? 0);

        internal Engine(string mode, string root, long[] ids, long[] attention, int dimensions, long budget)
        {
            this.dimensions = dimensions;
            Require(ExecutionOptions.Memory.Tensor.MaxDegreeOfParallelism == 1, "Managed execution must use one inference thread");
            var shape = new[] { 1, ids.Length };
            mask = new DenseTensor<long>(attention, shape);
            inputs = new Dictionary<string, ITensor> { ["input_ids"] = new DenseTensor<long>(ids, shape), ["attention_mask"] = mask };
            if (mode == "lok")
            {
                Require(CampaignEvidence.NativeModules().Length == 0, "Native ORT loaded in managed worker");
                prepared = OnnxImport.Load(Path.Combine(root, "prepared-fp32/model.onnx"), budget) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
                graph = prepared.CreateExecution(ExecutionOptions.Memory);
            }
            else
            {
                settings = NativeOptions();
                session = new InferenceSession(Path.Combine(root, "onnx/model.onnx"), settings);
                run = new RunOptions();
                native.Add("input_ids", OrtValue.CreateTensorValueFromMemory(ids, new long[] { 1, ids.Length }));
                native.Add("attention_mask", OrtValue.CreateTensorValueFromMemory(attention, new long[] { 1, ids.Length }));
                foreach (string name in new[] { "image_features", "video_features", "audio_features" })
                    native.Add(name, OrtValue.CreateTensorValueFromMemory(System.Array.Empty<float>(), new long[] { 0, 512 }));
            }
        }

        internal void Execute()
        {
            if (graph is not null)
            {
                if (!graph.Execute(inputs, true, ExecutionProvider.CPU, ExecutionOptions.Memory)) throw new InvalidDataException(graph.LastErrorMessage);
                Vector = EmbeddingGemma2Pooling.Pool((Tensor<float>)graph.Outputs["last_hidden_state"], mask, dimensions, CancellationToken.None)[0];
            }
            else
            {
                result = session!.Run(run!, native, outputNames);
                Vector = NativeVector(result.Single(), dimensions);
            }
        }

        internal string InputHash()
        {
            if (graph is not null) return CampaignEvidence.HashInputs(inputs);
            return CampaignEvidence.HashInputs(new Dictionary<string, ITensor> {
                ["input_ids"] = new DenseTensor<long>(native["input_ids"].GetTensorDataAsSpan<long>().ToArray(), inputs["input_ids"].Dims),
                ["attention_mask"] = new DenseTensor<long>(native["attention_mask"].GetTensorDataAsSpan<long>().ToArray(), inputs["attention_mask"].Dims) });
        }

        internal void Reset() { graph?.Reset(); result?.Dispose(); result = null; Vector = null; }

        internal object Profile(bool wallOnly)
        {
            if (graph is null) throw new InvalidOperationException("Only managed profiling is supported here");
            Reset();
            using var scope = wallOnly ? Profiler.BeginWallExecution() : Profiler.BeginExecution(true);
            long start = Stopwatch.GetTimestamp(); Execute(); long elapsed = Stopwatch.GetTimestamp() - start;
            var names = graph.Nodes.ToDictionary(n => n.ID, n => n.Name);
            object[] rows = wallOnly
                ? (graph.LastWallProfile ?? System.Array.Empty<WallNode>()).Select(n => (object)new {
                    node_id = n.NodeId, op = n.Op.ToString(), name = names[n.NodeId],
                    ms = (n.EndTicks - n.StartTicks) * 1000.0 / Stopwatch.Frequency }).ToArray()
                : (graph.LastProfile ?? new Stack<NodeProfile>()).Select(n => (object)new {
                    node_id = n.NodeId, op = n.Op.ToString(), name = names[n.NodeId], ms = n.OpsProfile.Sum(p => p.Time.TotalMilliseconds),
                    stages = n.OpsProfile.GroupBy(p => p.Stage.ToString()).ToDictionary(g => g.Key, g => g.Sum(p => p.Time.TotalMilliseconds)) }).ToArray();
            return new { mode = wallOnly ? "wall" : "stages", wall_ms = elapsed * 1000.0 / Stopwatch.Frequency, nodes = rows,
                graph.LastAllocatedBytes, graph.LastPoolAllocatedNewBytes, graph.LastPoolReusedBytes, graph.LastCopyBytes,
                graph.LastScratchBytes, graph.LastPeakLiveBytes, retained_packed_weight_bytes = PackedBytes };
        }
        public void Dispose()
        {
            Reset(); foreach (var input in native.Values) input.Dispose(); run?.Dispose(); session?.Dispose(); settings?.Dispose();
        }
    }

    static double Check(float[] actual, float[] reference)
    {
        Require(actual.Length == reference.Length, "Vector length differs");
        double worst = 0, delta = 0, dot = 0, normA = 0, normR = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double a = actual[i], r = reference[i], error = a - r;
            Require(double.IsFinite(a) && double.IsFinite(r), "Nonfinite vector");
            worst = Math.Max(worst, Math.Abs(error) / (1e-6 + 1e-5 * Math.Abs(r)));
            delta += error * error; dot += a * r; normA += a * a; normR += r * r;
        }
        Require(worst <= 1 && Math.Sqrt(delta / Math.Max(normR, 1e-24)) <= 1e-4 &&
            dot / Math.Sqrt(normA * normR) >= .99999 && Math.Abs(Math.Sqrt(normA) - 1) <= 1e-5, "Vector numerical agreement failed");
        return worst;
    }

    static void Run(string[] args)
    {
        if (!OperatingSystem.IsLinux() && !OperatingSystem.IsWindows()) throw new PlatformNotSupportedException("Affinity requires Linux or Windows");
        Require(args.Length == 2 && args[0] is "lok" or "ort" or "profile" or "lok-text" or "ort-text", "Usage: bench lok|ort|profile|lok-text|ort-text configuration.json");
        bool profiling = args[0] == "profile";
        string started = DateTime.UtcNow.ToString("O"), mode = profiling ? "lok" : args[0], configuration = Path.GetFullPath(args[1]);
        string configurationHash = CampaignEvidence.HashFile(configuration);
        var config = Read<Configuration>(configuration);
        Require(config.Cpu is >= 0 and < 64 && Cases.TryGetValue(config.Case, out _), "Unknown CPU/case");
        EmbeddingGemma2Pooling.ValidateDimensions(config.Dimensions);
        Require(config.PackedWeightBudget >= 0 && Regex.IsMatch(config.SourceSha, "^[a-f0-9]{40}$"), "Invalid source/budget");
        using var process = Process.GetCurrentProcess();
        long affinity = 1L << config.Cpu;
        Require(CurrentAffinity() == affinity && Environment.ProcessorCount == 1, "Confine the process before runtime startup");
        Require(!File.Exists(config.Output), "Output already exists");
        string core = typeof(ComputationalGraph).Assembly.Location;
        Require(CampaignEvidence.HashFile(core) == config.CoreSha256, "Loaded core differs");
        string fixturePath = Path.GetFullPath(config.Fixture);
        Require(CampaignEvidence.HashFile(fixturePath) == config.FixtureSha256, "Fixture manifest differs");
        using var fixture = JsonDocument.Parse(File.ReadAllBytes(fixturePath));
        var root = fixture.RootElement;
        Require(root.GetProperty("protocol").GetString() == Protocol, "Fixture protocol differs");
        string modelRoot = Path.Combine(config.Root, "models/embeddinggemma-2");
        var files = new Dictionary<string, string> { ["onnx/model.onnx"] = root.GetProperty("modelSha256").GetString()!,
            ["onnx/model.onnx_data"] = root.GetProperty("weightsSha256").GetString()!,
            ["prepared-fp32/model.onnx"] = root.GetProperty("preparedGraphSha256").GetString()!,
            ["prepared-fp32/model.onnx_data"] = root.GetProperty("weightsSha256").GetString()!,
            ["prepared-fp32/preparation.json"] = root.GetProperty("preparationSha256").GetString()!,
            ["tokenizer.json"] = root.GetProperty("tokenizerSha256").GetString()! };
        foreach (var file in files) Require(CampaignEvidence.HashFile(Path.Combine(modelRoot, file.Key)) == file.Value, "Changed model asset: " + file.Key);
        var item = root.GetProperty("cases").EnumerateArray().Single(c => c.GetProperty("id").GetString() == config.Case);
        var definition = Cases[config.Case];
        Require(item.GetProperty("sequenceLength").GetInt32() == definition.Width && item.GetProperty("unmaskedTokens").GetInt32() == definition.Real, "Fixture case shape differs");
        string directory = Path.Combine(Path.GetDirectoryName(fixturePath)!, config.Case);
        long[] ids = Array<long>(directory, item.GetProperty("input_ids"), definition.Width, "int64");
        long[] mask = Array<long>(directory, item.GetProperty("attention_mask"), definition.Width, "int64");
        Require(mask.Take(definition.Real).All(v => v == 1) && mask.Skip(definition.Real).All(v => v == 0) && ids.Skip(definition.Real).All(v => v == 0), "Incorrect right padding");
        float[] expected = Array<float>(directory, item.GetProperty("embedding_" + config.Dimensions), config.Dimensions, "float32");
        var runner = CampaignEvidence.RunnerIdentity();
        var host = CampaignEvidence.DescribeHost(affinity, CampaignEvidence.CpuIdentity());
        if (args[0] is "lok-text" or "ort-text")
        {
            TextPipeline(args[0], config, modelRoot, item.GetProperty("text").GetString()!, expected, runner.Hash, host);
            return;
        }
        TimeSpan cpuBefore = process.TotalProcessorTime;
        var confinementWatch = Stopwatch.StartNew(); double sink = 0;
        while (confinementWatch.ElapsedMilliseconds < (config.Smoke ? 100 : 2000))
            for (int i = 1; i <= 20000; i++) sink += Math.Sqrt(i);
        confinementWatch.Stop(); GC.KeepAlive(sink);
        process.Refresh(); double wall = confinementWatch.Elapsed.TotalMilliseconds, cpu = (process.TotalProcessorTime - cpuBefore).TotalMilliseconds;
        Require(config.Smoke || cpu / wall is >= .8 and <= 1.3, "CPU confinement counters failed");
        long load = Stopwatch.GetTimestamp();
        using var engine = new Engine(mode, modelRoot, ids, mask, config.Dimensions, config.PackedWeightBudget);
        long loadTicks = Stopwatch.GetTimestamp() - load;
        var nativeModules = CampaignEvidence.NativeModules();
        Require(mode == "lok" ? nativeModules.Length == 0 : nativeModules.Length == 1, "Native module count differs");
        if (mode == "ort") Require(OrtEnv.Instance().GetVersionString() == "1.29.0", "Expected pinned native ORT 1.29.0");
        string? nativeHash = mode == "lok" ? null : CampaignEvidence.HashFile(nativeModules.Single());
        object? nativeIdentity = mode == "lok" ? null : new { path = nativeModules.Single(), sha256 = nativeHash,
            architecture = RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant() };
        string inputHash = engine.InputHash();
        long firstStart = Stopwatch.GetTimestamp(); engine.Execute(); long firstTicks = Stopwatch.GetTimestamp() - firstStart;
        double pre = Check(engine.Vector!, expected); engine.Reset();
        var warm = new List<long>(); var warmMs = new List<double>(); double total = 0;
        var warmWall = Stopwatch.StartNew(); bool steady;
        do
        {
            engine.Reset(); long start = Stopwatch.GetTimestamp(); engine.Execute(); long ticks = Stopwatch.GetTimestamp() - start;
            engine.Reset(); warm.Add(ticks); warmMs.Add(ticks * 1000.0 / Stopwatch.Frequency); total += warmMs[^1];
            steady = config.Smoke || warm.Count >= 32 && total >= 1000 && CampaignEvidence.SteadyWindow(warmMs);
        } while (!steady && warm.Count < 1000 && warmWall.Elapsed.TotalSeconds < 60);
        warmWall.Stop();
        Require(steady && warmWall.Elapsed.TotalSeconds <= 60, "Warmup did not converge within 1000 calls/60 seconds");
        if (profiling)
        {
            var profiles = new List<object>();
            foreach (bool wallOnly in new[] { true, false })
            {
                profiles.Add(engine.Profile(wallOnly));
                Check(engine.Vector!, expected);
                engine.Reset();
            }
            using var profileOutput = new FileStream(config.Output, FileMode.CreateNew, FileAccess.Write);
            JsonSerializer.Serialize(profileOutput, new { scope = "diagnostic-only; unscored", @case = config.Case, dimensions = config.Dimensions,
                packed_weight_budget = config.PackedWeightBudget, source_sha = config.SourceSha, core_sha256 = config.CoreSha256,
                runner_sha256 = runner.Hash, fixture_sha256 = config.FixtureSha256, profiles }, Json);
            Console.WriteLine("PASS numerical agreement and managed wall/stage profiling (unscored)");
            return;
        }
        var blocks = new List<object>();
        for (int b = 0; b < (config.Smoke ? 1 : 3); b++)
        {
            var ticks = new long[config.Smoke ? 2 : 11];
            var before = SampleDiagnostics.Snapshot.Capture();
            long blockStart = Stopwatch.GetTimestamp();
            for (int c = 0; c < ticks.Length; c++)
            {
                engine.Reset(); long start = Stopwatch.GetTimestamp(); engine.Execute(); ticks[c] = Stopwatch.GetTimestamp() - start;
            }
            engine.Reset(); long blockTicks = Stopwatch.GetTimestamp() - blockStart;
            var after = SampleDiagnostics.Snapshot.Capture();
            blocks.Add(new { index = b, block_ticks = blockTicks, execute_ticks = ticks,
                allocated_bytes = after.AllocatedBytes - before.AllocatedBytes, process_cpu_ns = after.ProcessCpuNs - before.ProcessCpuNs,
                thread_cpu_ns = after.ThreadCpuNs - before.ThreadCpuNs, gen0 = after.Gen0 - before.Gen0, gen1 = after.Gen1 - before.Gen1,
                gen2 = after.Gen2 - before.Gen2, gc_pause_ticks = after.PauseTicks - before.PauseTicks });
        }
        long postStart = Stopwatch.GetTimestamp(); engine.Execute(); long postTicks = Stopwatch.GetTimestamp() - postStart;
        double post = Check(engine.Vector!, expected); engine.Reset();
        Require(engine.InputHash() == inputHash, "Inputs mutated");
        foreach (var file in files) Require(CampaignEvidence.HashFile(Path.Combine(modelRoot, file.Key)) == file.Value, "Model changed during measurement");
        Require(CampaignEvidence.RunnerIdentity().Hash == runner.Hash && CampaignEvidence.HashFile(core) == config.CoreSha256 &&
            CampaignEvidence.HashFile(configuration) == configurationHash && CampaignEvidence.HashFile(fixturePath) == config.FixtureSha256, "Worker files changed");
        process.Refresh();
        Require(CurrentAffinity() == affinity && (mode != "lok" || CampaignEvidence.NativeModules().Length == 0), "Worker isolation changed");
        if (mode == "ort") Require(CampaignEvidence.NativeModules().Single() == nativeModules.Single() &&
            CampaignEvidence.HashFile(nativeModules.Single()) == nativeHash, "Native binary changed");
        using var output = new FileStream(config.Output, FileMode.CreateNew, FileAccess.Write);
        JsonSerializer.Serialize(output, new { producer = Protocol, timing_contract = Contract, mode, @case = config.Case,
            smoke = config.Smoke, dimensions = config.Dimensions, packed_weight_budget = config.PackedWeightBudget,
            process_id = Environment.ProcessId, started_utc = started, completed_utc = DateTime.UtcNow.ToString("O"), exit_code = 0,
            source_sha = config.SourceSha, core_sha256 = config.CoreSha256, configuration_sha256 = configurationHash,
            runner_sha256 = runner.Hash, runner_files = runner.Files, environment = host, native_module = nativeIdentity,
            ort_version = mode == "ort" ? OrtEnv.Instance().GetVersionString() : null, fixture_sha256 = config.FixtureSha256,
            assets = files, input_sha256 = inputHash, inputs_intact = true, load_ticks = loadTicks,
            retained_packed_weight_bytes = engine.PackedBytes, peak_working_set_bytes = process.PeakWorkingSet64,
            stopwatch_frequency = Stopwatch.Frequency, confinement = new { wall_ms = wall, cpu_ms = cpu, ratio = cpu / wall },
            measured = new { pre_component_allowance_ratio = pre, post_component_allowance_ratio = post,
                first_execute_ticks = firstTicks, post_execute_ticks = postTicks, warmup_wall_ticks = warmWall.ElapsedTicks, warmup_ticks = warm,
                warmup_stop = config.Smoke ? "fixed-smoke" : "steady", blocks } }, Json);
        Console.WriteLine("PASS " + Protocol + " " + config.Case + " " + mode + (config.Smoke ? " (unscored smoke)" : ""));
    }

    static void TextPipeline(string mode, Configuration config, string modelRoot, string formatted, float[] expected,
        string runnerHash, Dictionary<string, object> host)
    {
        Require(config.Case != "gemma2-32pad128", "Text timing uses natural token lengths, not artificial padding");
        const string prefix = "title: Observations | text: ";
        Require(formatted.StartsWith(prefix, StringComparison.Ordinal), "Unexpected text fixture prompt");
        var document = new EmbeddingGemma2Document(formatted[prefix.Length..], "Observations");
        bool managed = mode == "lok-text";
        long load = Stopwatch.GetTimestamp();
        var api = managed ? new EmbeddingGemma2Embedder(Path.Combine(modelRoot, "prepared-fp32"),
            new EmbeddingGemma2Options { PreparedWeightBudgetBytes = config.PackedWeightBudget }) : null;
        using var settings = managed ? null : NativeOptions();
        using var session = managed ? null : new InferenceSession(Path.Combine(modelRoot, "onnx/model.onnx"), settings);
        using var run = managed ? null : new RunOptions();
        var tokenizer = managed ? null : new GemmaTokenizer(Path.Combine(modelRoot, "tokenizer.json"));
        var inputs = new Dictionary<string, OrtValue>();
        if (!managed) foreach (string name in new[] { "image_features", "video_features", "audio_features" })
            inputs.Add(name, OrtValue.CreateTensorValueFromMemory(System.Array.Empty<float>(), new long[] { 0, 512 }));
        long loadTicks = Stopwatch.GetTimestamp() - load;
        var names = new[] { "sentence_embedding" };
        float[] Infer()
        {
            if (api is not null) return api.EmbedDocuments(new[] { document }, config.Dimensions, CancellationToken.None)[0].Values;
            var batch = tokenizer!.EncodeBatch(new[] { EmbeddingGemma2Inputs.Document(document) }, 512);
            using var ids = OrtValue.CreateTensorValueFromMemory(batch.InputIds, new long[] { 1, batch.SequenceLength });
            using var mask = OrtValue.CreateTensorValueFromMemory(batch.AttentionMask, new long[] { 1, batch.SequenceLength });
            inputs["input_ids"] = ids; inputs["attention_mask"] = mask;
            using var result = session!.Run(run!, inputs, names);
            return NativeVector(result.Single(), config.Dimensions);
        }
        Check(Infer(), expected);
        var warm = new List<long>(); var warmMs = new List<double>();
        var warmWatch = Stopwatch.StartNew(); double total = 0; bool steady;
        do
        {
            long start = Stopwatch.GetTimestamp(); var vector = Infer(); long ticks = Stopwatch.GetTimestamp() - start;
            GC.KeepAlive(vector); warm.Add(ticks); warmMs.Add(ticks * 1000.0 / Stopwatch.Frequency); total += warmMs[^1];
            steady = total >= 1000 && CampaignEvidence.SteadyWindow(warmMs);
        } while (!steady && warm.Count < 1000 && warmWatch.Elapsed.TotalSeconds < 60);
        Require(steady, "Text pipeline warmup did not converge");
        var before = SampleDiagnostics.Snapshot.Capture();
        var raw = new long[33];
        for (int i = 0; i < raw.Length; i++)
        {
            long start = Stopwatch.GetTimestamp(); var vector = Infer(); raw[i] = Stopwatch.GetTimestamp() - start;
            GC.KeepAlive(vector);
        }
        var after = SampleDiagnostics.Snapshot.Capture();
        Check(Infer(), expected);
        foreach (string name in new[] { "image_features", "video_features", "audio_features" })
            if (inputs.TryGetValue(name, out var media)) media.Dispose();
        using var process = Process.GetCurrentProcess();
        using var output = new FileStream(config.Output, FileMode.CreateNew, FileAccess.Write);
        JsonSerializer.Serialize(output, new { scope = "diagnostic-only; complete text-to-owned-vector; unscored", mode,
            @case = config.Case, dimensions = config.Dimensions, packed_weight_budget = config.PackedWeightBudget,
            source_sha = config.SourceSha, core_sha256 = config.CoreSha256, runner_sha256 = runnerHash,
            fixture_sha256 = config.FixtureSha256, environment = host, load_ticks = loadTicks,
            stopwatch_frequency = Stopwatch.Frequency, warmup_ticks = warm, execute_ticks = raw,
            allocated_bytes = after.AllocatedBytes - before.AllocatedBytes, gen0 = after.Gen0 - before.Gen0,
            gen1 = after.Gen1 - before.Gen1, gen2 = after.Gen2 - before.Gen2, peak_working_set_bytes = process.PeakWorkingSet64 }, Json);
        Console.WriteLine("PASS complete text-to-vector numerical agreement and diagnostic timings (unscored)");
    }
}
