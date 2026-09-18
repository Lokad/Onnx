using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using Lokad.Onnx;
using Lokad.Onnx.Bench;
using Microsoft.ML.OnnxRuntime;

// One case and one engine per process. The oracle is another process entirely.
// This compiles against the released core; both arms run identical producer code.
internal static class IsolatedE5
{
    internal const string Protocol = "isolated-e5-v1";
    internal const string TimingContract = "public-execute-v1; all individual calls retained; reset/disposal outside; complete-request blocks separate";
    internal static readonly string[] Cases = ["e5-8tok", "e5-30tok", "e5-30pad128", "e5-128tok", "e5-512tok"];
    // This protocol targets the inspected, embedded-weight model. A different
    // export needs its own reviewed contract, including external-data identities.
    const string ModelHash = "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665";
    const double Tolerance = 1e-4;
    static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
        RespectRequiredConstructorParameters = true,
        WriteIndented = true
    };

    internal sealed record Configuration(string Root, string Case, int Cpu, string Output,
        string SourceSha, string CoreSha256, string Fixture, string? FixtureSha256, bool Smoke);
    internal sealed record NativeIdentity(string Path, string Sha256, string Architecture);
    internal sealed record OutputIdentity(string Name, int[] Dims, string Dtype, string File, string Sha256);
    internal sealed record Fixture(string Protocol, string Case, string ModelSha256, string TokenizerSha256,
        string InputSha256, int UnmaskedTokens, string OracleVersion, NativeIdentity Native, OutputIdentity[] Outputs);

    internal static T ReadJson<T>(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(path));
        void CheckKeys(JsonElement element)
        {
            if (element.ValueKind == JsonValueKind.Object)
            {
                var names = new HashSet<string>(StringComparer.Ordinal);
                foreach (var property in element.EnumerateObject())
                {
                    Require(names.Add(property.Name), "Duplicate JSON property: " + property.Name);
                    CheckKeys(property.Value);
                }
            }
            else if (element.ValueKind == JsonValueKind.Array)
                foreach (var item in element.EnumerateArray()) CheckKeys(item);
        }
        CheckKeys(document.RootElement);
        return document.RootElement.Deserialize<T>(Json) ?? throw new InvalidDataException("Empty JSON record");
    }

    internal static void WriteJson(string path, object value)
    {
        using var output = new FileStream(path, FileMode.CreateNew, FileAccess.Write);
        JsonSerializer.Serialize(output, value, Json);
    }

    static void Require([System.Diagnostics.CodeAnalysis.DoesNotReturnIf(false)] bool condition, string message)
    {
        if (!condition) throw new InvalidDataException(message);
    }

    static string Hash(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));

    static long CurrentAffinity()
    {
        if (OperatingSystem.IsLinux() || OperatingSystem.IsWindows())
        {
            using var process = Process.GetCurrentProcess();
            return process.ProcessorAffinity.ToInt64();
        }
        throw new PlatformNotSupportedException("Unsupported affinity platform");
    }

    internal static (int Tokens, int Unmasked, int Take, int Pad, string Text) Definition(string name) => name switch
    {
        "e5-8tok" => (8, 8, 0, 0, Bench.E5ShortText),
        "e5-30tok" => (30, 30, 0, 0, "query: " + Bench.E5Sentence),
        "e5-30pad128" => (128, 30, 0, 128, "query: " + Bench.E5Sentence),
        "e5-128tok" => (128, 128, 128, 0, "query: " + string.Join(" ", Enumerable.Repeat(Bench.E5Sentence, 40))),
        "e5-512tok" => (512, 512, 512, 0, "query: " + string.Join(" ", Enumerable.Repeat(Bench.E5Sentence, 40))),
        _ => throw new ArgumentException("Unknown isolated e5 case: " + name)
    };

    static Dictionary<string, ITensor> Inputs(string name, string tokenizer)
    {
        var definition = Definition(name);
        var inputs = Bench.E5Inputs(name, tokenizer, definition.Text, definition.Take, definition.Pad)
            .ToDictionary(t => t.Name, StringComparer.Ordinal);
        Require(inputs.Keys.Order(StringComparer.Ordinal).SequenceEqual(new[] { "attention_mask", "input_ids", "token_type_ids" }), "Input names differ");
        Require(inputs.Values.All(t => t is Tensor<long> && t.Dims.SequenceEqual(new[] { 1, definition.Tokens })), "Input shape/dtype differs");
        var mask = ((Tensor<long>)inputs["attention_mask"]).ToArray();
        Require(mask.All(v => v is 0 or 1) && mask.Sum() == definition.Unmasked, "Attention mask differs");
        return inputs;
    }

    static SessionOptions NativeOptions()
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            IntraOpNumThreads = 1, InterOpNumThreads = 1, ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
        };
        options.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
        options.AddSessionConfigEntry("session.inter_op.allow_spinning", "0");
        return options;
    }

    static NativeIdentity Native()
    {
        var modules = CampaignEvidence.NativeModules();
        Require(modules.Length == 1, "Exactly one loaded native ORT module is required");
        return new NativeIdentity(modules[0], CampaignEvidence.HashFile(modules[0]), RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant());
    }

    // This object owns the *only* inference engine in the worker. Merely referring
    // to managed ORT types does not initialize its native library; verify that
    // invariant both before and after every Lokad worker.
    sealed class Engine : IDisposable
    {
        readonly ComputationalGraph? graph;
        readonly InferenceSession? session;
        readonly SessionOptions? sessionOptions;
        readonly RunOptions? runOptions;
        readonly Dictionary<string, ITensor> inputs;
        readonly Dictionary<string, OrtValue> nativeInputs = new(StringComparer.Ordinal);
        readonly ExecutionOptions options = ExecutionOptions.Default;
        readonly string[] outputs;
        IDisposableReadOnlyCollection<OrtValue>? result;
        internal string[] Outputs => outputs;

        internal Engine(string mode, string model, Dictionary<string, ITensor> inputs)
        {
            this.inputs = inputs;
            Require(options.Tensor.MaxDegreeOfParallelism == 1, "Managed engine must be single-threaded");
            if (mode == "lok")
            {
                Require(CampaignEvidence.NativeModules().Length == 0, "ORT native library loaded before Lokad worker");
                graph = OnnxImport.Load(model) ?? throw new InvalidDataException("Model import failed");
                outputs = graph.OutputDescs.Select(t => t.Name).Order(StringComparer.Ordinal).ToArray();
            }
            else
            {
                sessionOptions = NativeOptions();
                session = new InferenceSession(model, sessionOptions);
                runOptions = new RunOptions();
                foreach (var pair in inputs)
                    nativeInputs.Add(pair.Key, OrtValue.CreateTensorValueFromMemory(((Tensor<long>)pair.Value).ToArray(), pair.Value.Dims.Select(d => (long)d).ToArray()));
                outputs = session.OutputMetadata.Keys.Order(StringComparer.Ordinal).ToArray();
            }
            Require(outputs.Length > 0 && outputs.Distinct(StringComparer.Ordinal).Count() == outputs.Length, "Invalid model outputs");
        }

        internal void Reset()
        {
            graph?.Reset();
            result?.Dispose();
            result = null;
        }

        internal void Execute()
        {
            if (graph != null)
            {
                if (!graph.Execute(inputs, true, ExecutionProvider.CPU, options)) throw new InvalidDataException(graph.LastErrorMessage);
            }
            else result = session!.Run(runOptions!, nativeInputs, outputs);
        }

        internal (int[] Dims, float[] Data) Output(int index)
        {
            if (graph != null)
            {
                Require(graph.Outputs[outputs[index]] is Tensor<float>, "Output dtype must be float32");
                var tensor = (Tensor<float>)graph.Outputs[outputs[index]];
                return (tensor.Dimensions.ToArray(), tensor.ToArray());
            }
            var value = result!.ElementAt(index);
            var shape = value.GetTensorTypeAndShape();
            Require(shape.ElementDataType == Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float, "Output dtype must be float32");
            return (shape.Shape.Select(d => checked((int)d)).ToArray(), value.GetTensorDataAsSpan<float>().ToArray());
        }

        internal string InputHash()
        {
            if (graph != null) return CampaignEvidence.HashInputs(inputs);
            // Verify the actual buffers passed to native Run, not just the
            // source arrays from which those buffers were constructed.
            var observed = nativeInputs.ToDictionary(p => p.Key, p => (ITensor)new DenseTensor<long>(
                p.Value.GetTensorDataAsSpan<long>().ToArray(), inputs[p.Key].Dims), StringComparer.Ordinal);
            return CampaignEvidence.HashInputs(observed);
        }

        public void Dispose()
        {
            Reset();
            foreach (var value in nativeInputs.Values) value.Dispose();
            runOptions?.Dispose(); session?.Dispose(); sessionOptions?.Dispose();
        }
    }

    internal static float[] ReadOutput(string directory, OutputIdentity output, int index)
    {
        Require(output.File == "output-" + index + ".f32", "Oracle output filename differs");
        Require(!string.IsNullOrEmpty(output.Name) && output.Dtype == "float32" && output.Dims != null, "Invalid oracle output metadata");
        long size = 1;
        foreach (int dimension in output.Dims)
        {
            Require(dimension >= 0, "Negative oracle output dimension");
            size = checked(size * dimension);
        }
        Require(size <= int.MaxValue / sizeof(float), "Oracle output is too large");
        byte[] bytes = File.ReadAllBytes(Path.Combine(directory, output.File));
        Require(bytes.Length == size * sizeof(float) && Hash(bytes) == output.Sha256, "Oracle output bytes/digest differ");
        Require(BitConverter.IsLittleEndian, "Oracle raw float32 encoding requires little endian");
        var values = MemoryMarshal.Cast<byte, float>(bytes).ToArray();
        Require(values.All(float.IsFinite), "Oracle has non-finite values");
        return values;
    }

    internal static int Run(string[] args)
    {
        Require(args.Length == 2 && args[0] is "oracle" or "lok" or "ort", "Usage: isolate oracle|lok|ort configuration.json");
        Require(CampaignEvidence.Current == null, "Isolated producer cannot use legacy evidence options");
        string started = DateTime.UtcNow.ToString("O"), mode = args[0];
        string configPath = Path.GetFullPath(args[1]);
        string configHash = CampaignEvidence.HashFile(configPath);
        var config = ReadJson<Configuration>(configPath);
        Require(config.Cpu is >= 0 and < 64, "Invalid CPU selector");
        Require(Regex.IsMatch(config.SourceSha ?? "", "^[0-9a-f]{40}$") && Regex.IsMatch(config.CoreSha256 ?? "", "^[0-9a-f]{64}$"), "Full lowercase source/core digests required");
        string output = Path.GetFullPath(config.Output), root = Path.GetFullPath(config.Root), fixtureDirectory = Path.GetFullPath(config.Fixture);
        Require(!File.Exists(output), "Process output already exists");
        if (mode == "oracle") Require(!Directory.Exists(fixtureDirectory) && config.FixtureSha256 == null, "Oracle fixture directory must be new and unhashed");
        else Require(Regex.IsMatch(config.FixtureSha256 ?? "", "^[0-9a-f]{64}$"), "Worker requires expected oracle manifest digest");
        Definition(config.Case);
        string corePath = typeof(ComputationalGraph).Assembly.Location;
        Require(CampaignEvidence.HashFile(corePath) == config.CoreSha256, "Loaded core differs from expected digest");
        long affinity = 1L << config.Cpu;
        Require(CurrentAffinity() == affinity && Environment.ProcessorCount == 1, "Supervisor must confine process before runtime startup");
        var host = CampaignEvidence.DescribeHost(affinity, CampaignEvidence.CpuIdentity());
        var runner = CampaignEvidence.RunnerIdentity();
        string model = Path.Combine(root, "models", "multilingual-e5-small", "model.onnx");
        string tokenizer = Path.Combine(root, "models", "multilingual-e5-small", "sentencepiece.bpe.model");
        Require(CampaignEvidence.HashFile(model) == ModelHash, "Isolated e5 protocol model differs from inspected export");
        string tokenizerHash = CampaignEvidence.HashFile(tokenizer);
        var inputs = Inputs(config.Case, tokenizer);
        string inputHash = CampaignEvidence.HashInputs(inputs);
        var definition = Definition(config.Case);
        string fixturePath = Path.Combine(fixtureDirectory, "fixture.json");
        Fixture? fixture = null;
        float[][]? expected = null;
        if (mode != "oracle")
        {
            Require(CampaignEvidence.HashFile(fixturePath) == config.FixtureSha256, "Oracle manifest digest differs");
            fixture = ReadJson<Fixture>(fixturePath);
            Require(fixture.Protocol == Protocol && fixture.Case == config.Case && fixture.ModelSha256 == ModelHash
                && fixture.TokenizerSha256 == tokenizerHash && fixture.InputSha256 == inputHash && fixture.UnmaskedTokens == definition.Unmasked, "Oracle workload identity differs");
            Require(fixture.Outputs != null && fixture.Outputs.Length > 0, "Oracle outputs missing");
            expected = fixture.Outputs.Select((value, index) => ReadOutput(fixtureDirectory, value, index)).ToArray();
        }
        var confinement = Bench.MeasureSingleCpuConfinement(config.Smoke ? 100 : 2000);
        Require(config.Smoke || confinement.ratio is >= 0.8 and <= 1.3, "Single-CPU confinement counters failed");
        long loadStart = Stopwatch.GetTimestamp();
        using var engine = new Engine(mode == "oracle" ? "ort" : mode, model, inputs);
        long loadTicks = Stopwatch.GetTimestamp() - loadStart;
        NativeIdentity? native = mode == "lok" ? null : Native();
        if (mode == "ort") Require(native!.Sha256 == fixture!.Native.Sha256 && OrtEnv.Instance().GetVersionString() == fixture.OracleVersion, "Worker native ORT differs from oracle");
        var measured = new Dictionary<string, object>();
        if (mode == "oracle")
        {
            Directory.CreateDirectory(fixtureDirectory);
            engine.Execute();
            var outputs = new List<OutputIdentity>();
            for (int i = 0; i < engine.Outputs.Length; i++)
            {
                var value = engine.Output(i);
                Require(value.Data.All(float.IsFinite), "Oracle produced non-finite outputs");
                byte[] bytes = MemoryMarshal.AsBytes(value.Data.AsSpan()).ToArray();
                string file = "output-" + i + ".f32";
                using (var stream = new FileStream(Path.Combine(fixtureDirectory, file), FileMode.CreateNew, FileAccess.Write)) stream.Write(bytes);
                outputs.Add(new OutputIdentity(engine.Outputs[i], value.Dims, "float32", file, Hash(bytes)));
            }
            fixture = new Fixture(Protocol, config.Case, ModelHash, tokenizerHash, inputHash, definition.Unmasked,
                OrtEnv.Instance().GetVersionString(), native!, outputs.ToArray());
            WriteJson(fixturePath, fixture);
        }
        else
        {
            Require(engine.Outputs.SequenceEqual(fixture!.Outputs.Select(o => o.Name)), "Oracle/model output set or order differs");
            (double Error, long Ticks) Validate()
            {
                engine.Reset();
                long start = Stopwatch.GetTimestamp(); engine.Execute();
                long ticks = Stopwatch.GetTimestamp() - start;
                double worst = 0;
                for (int i = 0; i < engine.Outputs.Length; i++)
                {
                    var value = engine.Output(i);
                    var agreement = BenchValidate.RequireAgreement(engine.Outputs[i], fixture.Outputs[i].Dims, expected![i], value.Dims, value.Data, Tolerance);
                    worst = Math.Max(worst, agreement.scaled);
                }
                engine.Reset();
                return (worst, ticks);
            }
            var pre = Validate();
            var warmTicks = new List<long>();
            var warmMs = new List<double>();
            var warmWatch = Stopwatch.StartNew();
            double accumulated = 0;
            bool steady = false;
            do
            {
                engine.Reset();
                long start = Stopwatch.GetTimestamp(); engine.Execute();
                long ticks = Stopwatch.GetTimestamp() - start;
                engine.Reset();
                warmTicks.Add(ticks); double ms = ticks * 1000.0 / Stopwatch.Frequency;
                warmMs.Add(ms); accumulated += ms;
                steady = config.Smoke || (accumulated >= 1000 && CampaignEvidence.SteadyWindow(warmMs));
            } while (!steady && warmTicks.Count < 1000 && warmWatch.Elapsed.TotalSeconds < 60);
            Require(steady, "Warmup failed to converge within 1000 calls/60 seconds; no timed result");
            // Three blocks of eleven preserve the existing 33-call contract.
            // Block counters are outside the block stopwatch; every Execute/Run
            // stopwatch is retained and remains the sole primary metric.
            int blockCount = config.Smoke ? 1 : 3, calls = config.Smoke ? 2 : 11;
            var raw = new long[blockCount][];
            for (int i = 0; i < blockCount; i++) raw[i] = new long[calls];
            var blocks = new List<object>(blockCount);
            for (int b = 0; b < blockCount; b++)
            {
                var before = SampleDiagnostics.Snapshot.Capture();
                long blockStart = Stopwatch.GetTimestamp();
                for (int c = 0; c < calls; c++)
                {
                    engine.Reset();
                    long start = Stopwatch.GetTimestamp(); engine.Execute();
                    raw[b][c] = Stopwatch.GetTimestamp() - start;
                }
                engine.Reset();
                long blockTicks = Stopwatch.GetTimestamp() - blockStart;
                var after = SampleDiagnostics.Snapshot.Capture();
                blocks.Add(new
                {
                    index = b, block_ticks = blockTicks, execute_ticks = raw[b],
                    allocated_bytes = after.AllocatedBytes - before.AllocatedBytes,
                    thread_cpu_ns = after.ThreadCpuNs - before.ThreadCpuNs,
                    process_cpu_ns = after.ProcessCpuNs - before.ProcessCpuNs,
                    gen0 = after.Gen0 - before.Gen0, gen1 = after.Gen1 - before.Gen1, gen2 = after.Gen2 - before.Gen2,
                    gc_pause_ticks = after.PauseTicks - before.PauseTicks
                });
            }
            var post = Validate();
            measured.Add("pre_scaled_error", pre.Error); measured.Add("post_scaled_error", post.Error);
            measured.Add("first_execute_ticks", pre.Ticks); measured.Add("post_execute_ticks", post.Ticks);
            measured.Add("warmup_ticks", warmTicks); measured.Add("warmup_stop", config.Smoke ? "fixed-smoke" : "steady");
            measured.Add("blocks", blocks);
        }
        Require(engine.InputHash() == inputHash && CampaignEvidence.HashInputs(inputs) == inputHash, "Inference mutated inputs");
        engine.Reset();
        Require(CurrentAffinity() == affinity, "Worker affinity changed");
        Require(CampaignEvidence.HashFile(corePath) == config.CoreSha256 && CampaignEvidence.HashFile(model) == ModelHash
            && CampaignEvidence.HashFile(tokenizer) == tokenizerHash && CampaignEvidence.HashFile(configPath) == configHash, "Worker files changed");
        Require(CampaignEvidence.RunnerIdentity().Hash == runner.Hash, "Runner files changed");
        string fixtureHash = CampaignEvidence.HashFile(fixturePath);
        Require(mode == "oracle" || fixtureHash == config.FixtureSha256, "Oracle manifest changed");
        for (int i = 0; i < fixture!.Outputs.Length; i++) ReadOutput(fixtureDirectory, fixture.Outputs[i], i);
        if (mode == "lok") Require(CampaignEvidence.NativeModules().Length == 0, "Native ORT loaded in Lokad worker");
        else Require(Native() == native, "Native ORT changed");
        WriteJson(output, new
        {
            producer = Protocol, timing_contract = TimingContract, mode, @case = config.Case, smoke = config.Smoke,
            process_id = Environment.ProcessId, started_utc = started, completed_utc = DateTime.UtcNow.ToString("O"), exit_code = 0,
            configuration_sha256 = configHash, source_sha = config.SourceSha, core_sha256 = config.CoreSha256, core_path = corePath,
            runner_sha256 = runner.Hash, runner_files = runner.Files, environment = host, native_module = native,
            fixture_sha256 = fixtureHash, model_sha256 = ModelHash, tokenizer_sha256 = tokenizerHash, input_sha256 = inputHash,
            external_data = new Dictionary<string, string>(), unmasked_tokens = definition.Unmasked, inputs_intact = true,
            ort_version = fixture.OracleVersion, oracle_native_sha256 = fixture.Native.Sha256,
            execution = new { managed = "auto", threads = 1, ort_provider = "cpu-only", ort_optimizations = "ORT_ENABLE_ALL", intraop = 1, interop = 1, spinning = false, sequential = true },
            confinement = new { wall_ms = confinement.wallMs, cpu_ms = confinement.cpuMs, ratio = confinement.ratio },
            stopwatch_frequency = Stopwatch.Frequency, load_ticks = loadTicks, measured
        });
        Console.WriteLine(Protocol + " " + mode + " " + config.Case + " complete; inputs intact; output=" + output);
        return 0;
    }
}
