using System.Diagnostics;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

if (args.Length != 3) throw new ArgumentException("Usage: TranscribeReplay <Parakeet-directory> <manifest.json> <new-result.json>");
string models = Path.GetFullPath(args[0]), manifestPath = Path.GetFullPath(args[1]), destination = Path.GetFullPath(args[2]);
string fixtures = Path.GetDirectoryName(manifestPath) ?? throw new InvalidDataException("Missing fixture directory");
string tensorRoot = destination + ".tensors";
if (File.Exists(destination) || Directory.Exists(tensorRoot)) throw new IOException("Output already exists");
var json = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower, WriteIndented = true };
json.Converters.Add(new JsonStringEnumConverter());
var rows = new List<object>(); var errors = new List<string>();
bool passed = true; int comparisons = 0, rejections = 0; long valuesCompared = 0;
double worst = 0;
string Sha(string path) { using var stream = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(stream)); }
string SourceSha(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(Encoding.UTF8.GetString(bytes).Replace("\r\n", "\n", StringComparison.Ordinal))));
string Text(JsonElement value) => value.GetString() ?? throw new InvalidDataException("Missing string");
void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
string Below(string root, string name)
{
    string path = Path.GetFullPath(Path.Combine(root, name));
    Require(path.StartsWith(root.TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar,
        OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal), "Path escapes fixture root");
    return path;
}
byte[] Bytes(ITensor value) => value switch
{
    Tensor<float> f => MemoryMarshal.AsBytes(f.ToArray().AsSpan()).ToArray(),
    Tensor<int> i => MemoryMarshal.AsBytes(i.ToArray().AsSpan()).ToArray(),
    Tensor<long> l => MemoryMarshal.AsBytes(l.ToArray().AsSpan()).ToArray(),
    _ => throw new InvalidDataException("Unsupported tensor dtype")
};
string Bits(ITensor value) => Convert.ToHexStringLower(SHA256.HashData(Bytes(value)));
void NoNative()
{
    using var process = Process.GetCurrentProcess();
    foreach (ProcessModule module in process.Modules)
        Require(!module.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase), "Native ORT loaded into managed replay");
}
object Private(object value, string field) => value.GetType().GetField(field, BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(value)
    ?? throw new InvalidDataException("Missing private field: " + field);
var tensors = new Dictionary<string, ITensor>();
var watch = Stopwatch.StartNew();
try
{
    NoNative();
    using var document = JsonDocument.Parse(File.ReadAllText(manifestPath));
    var manifest = document.RootElement;
    Require(manifest.GetProperty("schema").GetInt32() == 1 && Text(manifest.GetProperty("scope")) == "parakeet-transcription"
        && manifest.GetProperty("scaled_absolute_tolerance").GetDouble() == 1e-4, "Unknown scope or numerical gate");
    Require(Text(manifest.GetProperty("numpy")) == "2.2.4" && Text(manifest.GetProperty("onnxruntime")) == "1.29.0", "Native versions differ");
    using var assetResource = Assembly.GetExecutingAssembly().GetManifestResourceStream("parakeet-transcription-assets.json") ?? throw new InvalidDataException();
    using var assetBytes = new MemoryStream(); assetResource.CopyTo(assetBytes);
    using var assetDocument = JsonDocument.Parse(assetBytes.ToArray());
    Require(JsonNode.DeepEquals(JsonNode.Parse(assetDocument.RootElement.GetRawText()), JsonNode.Parse(manifest.GetProperty("assets").GetRawText())), "Asset pins differ");
    Require(SourceSha(assetBytes.ToArray()) == Text(manifest.GetProperty("assets_lf_sha256")), "Asset source differs");
    using var generator = Assembly.GetExecutingAssembly().GetManifestResourceStream("parakeet-transcription-generator.py") ?? throw new InvalidDataException();
    using var generatorBytes = new MemoryStream(); generator.CopyTo(generatorBytes);
    Require(SourceSha(generatorBytes.ToArray()) == Text(manifest.GetProperty("generator_lf_sha256")), "Native generator differs");
    foreach (var item in assetDocument.RootElement.GetProperty("files").EnumerateObject())
    {
        string path = Below(models, item.Name);
        Require(new FileInfo(path).Length == item.Value.GetProperty("bytes").GetInt64() && Sha(path) == Text(item.Value.GetProperty("sha256")), "Model asset differs: " + item.Name);
    }
    var settings = manifest.GetProperty("native_settings");
    Require(Text(settings.GetProperty("provider")) == "CPUExecutionProvider" && settings.GetProperty("threads").GetInt32() == 1
        && Text(settings.GetProperty("execution")) == "sequential" && Text(settings.GetProperty("optimization")) == "all"
        && !settings.GetProperty("spinning").GetBoolean(), "Native settings differ");
    foreach (var item in manifest.GetProperty("files").EnumerateObject())
    {
        string path = Below(fixtures, item.Name);
        Require(new FileInfo(path).Length == item.Value.GetProperty("bytes").GetInt64() && Sha(path) == Text(item.Value.GetProperty("sha256")), "Fixture digest differs: " + item.Name);
        var tensor = NpySupport.ReadTensor(path);
        string dtype = tensor is Tensor<float> ? "float32" : tensor is Tensor<int> ? "int32" : tensor is Tensor<long> ? "int64" : "unsupported";
        Require(dtype == Text(item.Value.GetProperty("dtype")) && tensor.Dims.SequenceEqual(item.Value.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32())), "Fixture metadata differs");
        if (tensor is Tensor<float> f) Require(f.ToArray().All(float.IsFinite), "Nonfinite fixture");
        tensors.Add(item.Name, tensor);
    }
    var cases = manifest.GetProperty("cases").EnumerateArray().ToArray();
    string[] coverage = ["english-16k", "french-44k-stereo", "jfk-48k-stereo", "english-token-limit", "english-frame-limit", "silence", "english-repeat"];
    Require(cases.Select(c => Text(c.GetProperty("name"))).SequenceEqual(coverage), "Case coverage differs");
    for (int i = 0; i < cases.Length; i++)
    {
        Require(cases[i].GetProperty("max_tokens").GetInt32() == (i == 3 ? 3 : 4096)
            && cases[i].GetProperty("max_tokens_per_frame").GetInt32() == (i == 4 ? 1 : 10), "Policy coverage differs");
        Require(cases[i].GetProperty("stages").GetArrayLength() == (i == 5 ? 0 : 2), "Stage coverage differs");
        // Audit the oracle's recurrence before loading large model weights. In
        // particular, a blank cannot replace either incoming recurrent state.
        var item = cases[i];
        var pcm = tensors[Text(item.GetProperty("pcm"))];
        Require(pcm is Tensor<float> && pcm.Dims.Length == 1 && pcm.Length <= 480000, "PCM contract differs");
        int frames = i == 5 ? 0 : checked((int)((pcm.Length / 160 + 8) / 8));
        int frame = 0, emitted = 0;
        var nativeTokens = new List<int>(); var nativeFrames = new List<int>(); var nativeDurations = new List<int>();
        ITensor state1 = new DenseTensor<float>(new[] { 2, 1, 640 }), state2 = new DenseTensor<float>(new[] { 2, 1, 640 });
        foreach (var step in item.GetProperty("steps").EnumerateArray())
        {
            Require(frame < frames && nativeTokens.Count < item.GetProperty("max_tokens").GetInt32(), "Native trace continued past stop");
            Require(step.GetProperty("frame").GetInt32() == frame && step.GetProperty("target").GetInt32() == (nativeTokens.Count == 0 ? 8192 : nativeTokens[^1]), "Native trace target/frame differs");
            Require(step.GetProperty("state_input_sha256").GetArrayLength() == 2
                && Text(step.GetProperty("state_input_sha256")[0]) == Bits(state1)
                && Text(step.GetProperty("state_input_sha256")[1]) == Bits(state2), "Native recurrent state binding differs");
            var logits = tensors[Text(step.GetProperty("outputs").GetProperty("outputs"))] as Tensor<float> ?? throw new InvalidDataException();
            Require(logits.Dimensions.SequenceEqual(new[] { 1, 1, 1, 8198 }), "Native logits shape differs");
            var values = logits.ToArray();
            int token = 0, duration = 0;
            for (int j = 1; j < 8193; j++) if (values[j] > values[token]) token = j;
            for (int j = 1; j < 5; j++) if (values[8193 + j] > values[8193 + duration]) duration = j;
            Require(token == step.GetProperty("token").GetInt32() && duration == step.GetProperty("duration").GetInt32(), "Native decision differs from logits");
            if (token != 8192)
            {
                state1 = tensors[Text(step.GetProperty("outputs").GetProperty("output_states_1"))];
                state2 = tensors[Text(step.GetProperty("outputs").GetProperty("output_states_2"))];
                nativeTokens.Add(token); nativeFrames.Add(frame); nativeDurations.Add(duration); emitted++;
            }
            if (duration > 0) { frame += duration; emitted = 0; }
            else if (token == 8192 || emitted == item.GetProperty("max_tokens_per_frame").GetInt32()) { frame++; emitted = 0; }
        }
        var expected = item.GetProperty("expected");
        Require(nativeTokens.SequenceEqual(expected.GetProperty("token_ids").EnumerateArray().Select(v => v.GetInt32()))
            && nativeFrames.SequenceEqual(expected.GetProperty("frame_indices").EnumerateArray().Select(v => v.GetInt32()))
            && nativeDurations.SequenceEqual(expected.GetProperty("duration_frames").EnumerateArray().Select(v => v.GetInt32())), "Native result differs from trace");
        Require(expected.GetProperty("encoded_frames").GetInt32() == frames
            && expected.GetProperty("decoder_calls").GetInt32() == item.GetProperty("steps").GetArrayLength(), "Native length/call count differs");
        Require(Text(expected.GetProperty("stop_reason")) == (i == 5 ? "SilentInput" : frame >= frames ? "EndOfAudio" : "TokenLimit"), "Native stop differs");
        if (i != 5) Require(frame >= frames || nativeTokens.Count == item.GetProperty("max_tokens").GetInt32(), "Native trace stopped early");
    }
    Directory.CreateDirectory(tensorRoot);
    var model = new ParakeetTranscriber(models);
    var frontend = (ComputationalGraph)Private(model, "frontend");
    var encoder = (ComputationalGraph)Private(model, "encoder");
    var decoder = (ComputationalGraph)Private(model, "decoder");
    object generation = Private(model, "generation");
    var decodeMethod = generation.GetType().GetMethod("Decode", BindingFlags.Instance | BindingFlags.NonPublic) ?? throw new InvalidDataException();
    var heldResults = new List<(ParakeetTranscription Value, string Json)>();
    foreach (var item in cases)
    {
        string name = Text(item.GetProperty("name"));
        Console.WriteLine("Starting " + name);
        var pcm = ((Tensor<float>)tensors[Text(item.GetProperty("pcm"))]).ToArray();
        byte[] originalPcm = MemoryMarshal.AsBytes(pcm.AsSpan()).ToArray();
        var options = new ParakeetTranscriptionOptions(item.GetProperty("max_tokens").GetInt32(), item.GetProperty("max_tokens_per_frame").GetInt32());
        var compared = new List<object>();
        var held = new List<(ITensor Value, string Hash)>();
        void Keep(ITensor tensor) => held.Add((tensor, Bits(tensor)));
        void Compare(string label, IReadOnlyDictionary<string, ITensor> actual, JsonElement expected)
        {
            Require(actual.Keys.Order().SequenceEqual(expected.EnumerateObject().Select(p => p.Name).Order()), "Output names differ");
            foreach (var output in expected.EnumerateObject())
            {
                var a = actual[output.Name]; var b = tensors[Text(output.Value)];
                Require(a.ElementType == b.ElementType && a.Dims.SequenceEqual(b.Dims), "Output shape/type differs");
                double error = 0; int at = 0;
                if (a is Tensor<float> af && b is Tensor<float> bf)
                {
                    float[] av = af.ToArray(), bv = bf.ToArray();
                    Require(av.All(float.IsFinite), "Nonfinite managed output");
                    for (int j = 0; j < av.Length; j++)
                    {
                        double difference = Math.Abs((double)av[j] - bv[j]) / Math.Max(1, Math.Abs((double)bv[j]));
                        if (difference > error) { error = difference; at = j; }
                    }
                    valuesCompared += av.Length;
                }
                else { Require(Bytes(a).SequenceEqual(Bytes(b)), "Integer output differs"); valuesCompared += a.Length; }
                worst = Math.Max(worst, error); passed &= error <= 1e-4; comparisons++;
                byte[] bytes = Bytes(a); string file = name + "-" + label + "-" + output.Name + ".bin";
                using (var stream = new FileStream(Path.Combine(tensorRoot, file), FileMode.CreateNew, FileAccess.Write)) stream.Write(bytes);
                compared.Add(new { label, output = output.Name, file, shape = a.Dims, dtype = a.ElementType.ToString(), sha256 = Bits(a), max_error = error, worst_index = at, passed = error <= 1e-4 });
                Keep(a);
            }
        }
        IReadOnlyDictionary<string, ITensor> Execute(GraphExecution context, Dictionary<string, ITensor> feeds)
        {
            foreach (var value in feeds.Values) Keep(value);
            context.Reset();
            Require(context.Execute(feeds, true, ExecutionProvider.CPU, ExecutionOptions.Memory), context.LastErrorMessage ?? "Graph execution failed");
            return context.Outputs.ToDictionary(p => p.Key, p => p.Value ?? throw new InvalidDataException("Missing graph output"));
        }
        ParakeetTranscription? traced = null;
        if (item.GetProperty("stages").GetArrayLength() > 0)
        {
            var pre = frontend.CreateExecution(ExecutionOptions.Memory);
            var enc = encoder.CreateExecution(ExecutionOptions.Memory);
            var dec = decoder.CreateExecution(ExecutionOptions.Memory);
            try
            {
                var features = Execute(pre, new() { ["waveforms"] = new DenseTensor<float>(pcm, new[] { 1, pcm.Length }), ["waveforms_lens"] = new DenseTensor<long>(new[] { (long)pcm.Length }, new[] { 1 }) });
                Compare("frontend", features, item.GetProperty("stages")[0].GetProperty("outputs"));
                var encoded = Execute(enc, new() { ["audio_signal"] = features["features"], ["length"] = features["features_lens"] });
                Compare("encoder", encoded, item.GetProperty("stages")[1].GetProperty("outputs"));
                int frames = checked((int)((Tensor<long>)encoded["encoded_lengths"]).ToArray()[0]);
                int step = 0;
                ITensor carried1 = new DenseTensor<float>(new[] { 2, 1, 640 }), carried2 = new DenseTensor<float>(new[] { 2, 1, 640 });
                Func<Dictionary<string, ITensor>, IReadOnlyDictionary<string, ITensor>> callback = feeds =>
                {
                    Require(step < item.GetProperty("steps").GetArrayLength(), "Managed decoder used more calls than reference");
                    var expected = item.GetProperty("steps")[step];
                    Require(((Tensor<int>)feeds["targets"]).ToArray()[0] == expected.GetProperty("target").GetInt32(), "Managed target path diverged");
                    Require(Bits(feeds["input_states_1"]) == Bits(carried1) && Bits(feeds["input_states_2"]) == Bits(carried2), "Managed recurrence did not carry its own accepted states");
                    int frame = expected.GetProperty("frame").GetInt32();
                    var current = ((Tensor<float>)feeds["encoder_outputs"]).ToArray();
                    var hidden = ((Tensor<float>)encoded["outputs"]).ToArray();
                    for (int channel = 0; channel < 1024; channel++) Require(current[channel] == hidden[channel * frames + frame], "Managed frame path diverged");
                    var outputs = Execute(dec, feeds); Compare("step-" + step++, outputs, expected.GetProperty("outputs"));
                    var actualLogits = ((Tensor<float>)outputs["outputs"]).ToArray();
                    int chosen = 0;
                    for (int j = 1; j < 8193; j++) if (actualLogits[j] > actualLogits[chosen]) chosen = j;
                    if (chosen != 8192) { carried1 = outputs["output_states_1"]; carried2 = outputs["output_states_2"]; }
                    return outputs;
                };
                try { traced = (ParakeetTranscription)(decodeMethod.Invoke(generation, new object[] { encoded["outputs"], frames, options, callback, CancellationToken.None }) ?? throw new InvalidDataException()); }
                catch (TargetInvocationException ex) when (ex.InnerException is not null) { ExceptionDispatchInfo.Capture(ex.InnerException).Throw(); throw; }
                Require(step == item.GetProperty("steps").GetArrayLength(), "Managed decoder omitted reference calls");
            }
            finally { pre.Reset(); enc.Reset(); dec.Reset(); }
        }
        var actual = model.Transcribe(pcm, 16000, options, CancellationToken.None);
        string actualJson = JsonSerializer.Serialize(actual, json);
        Require(JsonNode.DeepEquals(JsonNode.Parse(actualJson), JsonNode.Parse(item.GetProperty("expected").GetRawText())), "Public transcription differs: " + actualJson);
        Require(traced is null || JsonSerializer.Serialize(traced, json) == actualJson, "Traced and public API decisions differ");
        Require(MemoryMarshal.AsBytes(pcm.AsSpan()).SequenceEqual(originalPcm), "Caller PCM changed");
        foreach (var tensor in held) Require(Bits(tensor.Value) == tensor.Hash, "Held tensor changed after later calls/reset");
        foreach (var previous in heldResults) Require(JsonSerializer.Serialize(previous.Value, json) == previous.Json, "Held transcription changed");
        heldResults.Add((actual, actualJson));
        rows.Add(new { name, actual, comparisons = compared });
        Console.WriteLine(name + " passed decisions; cumulative tensor error " + worst.ToString("R"));
    }
    void Reject(Action action, Type type)
    {
        try { action(); }
        catch (Exception ex) when (ex.GetType() == type) { rejections++; return; }
        throw new InvalidDataException("Expected rejection missing: " + type.Name);
    }
    Reject(() => model.Transcribe(new[] { float.NaN }, 16000, ParakeetTranscriptionOptions.Default, CancellationToken.None), typeof(ArgumentException));
    Reject(() => model.Transcribe(new float[480001], 16000, ParakeetTranscriptionOptions.Default, CancellationToken.None), typeof(ArgumentOutOfRangeException));
    Reject(() => model.Transcribe(new[] { 1f }, 16000, ParakeetTranscriptionOptions.Default, CancellationToken.None), typeof(ArgumentOutOfRangeException));
    Reject(() => model.Transcribe(Array.Empty<float>(), 8000, ParakeetTranscriptionOptions.Default, CancellationToken.None), typeof(ArgumentOutOfRangeException));
    Reject(() => model.Transcribe(Array.Empty<float>(), 16000, new(0, 10), CancellationToken.None), typeof(ArgumentOutOfRangeException));
    using (var canceled = new CancellationTokenSource())
    {
        canceled.Cancel();
        Reject(() => model.Transcribe(Array.Empty<float>(), 16000, ParakeetTranscriptionOptions.Default, canceled.Token), typeof(OperationCanceledException));
    }
    var firstPcm = ((Tensor<float>)tensors[Text(cases[0].GetProperty("pcm"))]).ToArray();
    var recovery = model.Transcribe(firstPcm, 16000, ParakeetTranscriptionOptions.Default, CancellationToken.None);
    Require(JsonSerializer.Serialize(recovery, json) == heldResults[0].Json, "Request after failure/cancellation differs");
    NoNative();
}
catch (Exception ex) { passed = false; errors.Add(ex.ToString()); Console.Error.WriteLine(ex); }
var result = new { scope = "Parakeet short-recording transcription and independently carried full-array replay", passed,
    application_passed = errors.Count == 0 && rows.Count == 7 && rejections == 6, comparisons,
    values_compared = valuesCompared, max_error = worst, rejections, seconds = watch.Elapsed.TotalSeconds,
    manifest_sha256 = Sha(manifestPath), core_sha256 = Sha(typeof(ComputationalGraph).Assembly.Location),
    data_sha256 = Sha(typeof(ParakeetTranscriber).Assembly.Location), runner_sha256 = Sha(Assembly.GetExecutingAssembly().Location),
    runtime = RuntimeInformation.FrameworkDescription, peak_working_set = Process.GetCurrentProcess().PeakWorkingSet64,
    settings = Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>()
        .Where(p => p.Key is string key && (key.StartsWith("LOKAD_ONNX_", StringComparison.OrdinalIgnoreCase)
            || key.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || key.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)))
        .ToDictionary(p => (string)p.Key, p => p.Value?.ToString()), rows, errors };
using (var stream = new FileStream(destination, FileMode.CreateNew, FileAccess.Write)) JsonSerializer.Serialize(stream, result, json);
Console.WriteLine($"Parakeet transcription passed={passed}; comparisons={comparisons}; max_error={worst:R}; rejected={rejections}");
return passed ? 0 : 1;
