using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;
using System.Text.Json;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using System.Runtime.InteropServices;
var evidence = new Evidence(args); string source = evidence.Reference, destination = evidence.Output, output = evidence.Output;
Directory.CreateDirectory(output);
object Call(string type, string name, object? instance, params object[] values)
{
    var t = typeof(Community1Diarizer).Assembly.GetType("Lokad.Onnx." + type, true) ?? throw new Exception(type);
    try { return t.GetMethod(name, BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance)?.Invoke(instance, values) ?? throw new Exception(name); }
    catch (TargetInvocationException e) when (e.InnerException is not null) { ExceptionDispatchInfo.Capture(e.InnerException).Throw(); throw; }
}
var seg = OnnxImport.Load(evidence.Models["segmentation"], 32L * 1024 * 1024) ?? throw new Exception();
var enc = OnnxImport.Load(evidence.Models["encoder"], 64L * 1024 * 1024) ?? throw new Exception();
var embedder = new WeSpeakerEmbedder(evidence.Models["encoder"], evidence.Models["projection"]);
var clusterer = new Community1Clusterer(evidence.Models["plda"]);
var reports = new List<object>();
void Save(string name, string stage, string reference, float[] values)
{
    string file = reports.Count.ToString("D3") + ".f32"; var bytes = MemoryMarshal.AsBytes(values.AsSpan()).ToArray(); File.WriteAllBytes(Path.Combine(output, file), bytes);
    reports.Add(new { name, stage, reference, file, length = values.Length, sha256 = Convert.ToHexStringLower(SHA256.HashData(bytes)) });
}
Tensor<float> Execute(GraphExecution execution, string name, Tensor<float> input, string result)
{
    execution.Reset();
    if (!execution.Execute(new Dictionary<string, ITensor> { [name] = input }, true, ExecutionProvider.CPU, ExecutionOptions.Memory)) throw new Exception(execution.LastErrorMessage);
    return execution.Outputs[result] as Tensor<float> ?? throw new Exception(result);
}
using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(source, "manifest.json")));
var segmentation = seg.CreateExecution(ExecutionOptions.Memory); var encoding = enc.CreateExecution(ExecutionOptions.Memory);
try
{
    foreach (var c in doc.RootElement.GetProperty("cases").EnumerateArray())
    {
        string name = c.GetProperty("name").GetString() ?? ""; var pcm = NpySupport.ReadFloat32(Path.Combine(source, c.GetProperty("pcm").GetString() ?? "")).Values;
        int chunks = c.GetProperty("windows").GetArrayLength(); var activity = new bool[chunks * 589 * 3]; var embeddings = new WeSpeakerEmbedding[chunks * 3];
        for (int i = 0; i < chunks; i++)
        {
            var window = c.GetProperty("windows")[i]; var samples = (float[])Call("Community1Timeline", "Window", null, pcm, i);
            var scores = Execute(segmentation, "waveform", new DenseTensor<float>(samples, new[] { 1, 1, 160000 }), "scores").ToArray();
            Save(name, "scores", window.GetProperty("scores").GetString() ?? "", scores);
            var a = (bool[])Call("Community1Timeline", "Powerset", null, (object)scores); Array.Copy(a, 0, activity, i * a.Length, a.Length);
            Save(name, "activity", window.GetProperty("activity").GetString() ?? "", a.Select(v => v ? 1f : 0).ToArray());
            if (!window.TryGetProperty("features", out var feature)) continue;
            var masks = (float[][])Call("Community1Timeline", "EmbeddingMasks", null, (object)a);
            Save(name, "masks", window.GetProperty("masks").GetString() ?? "", masks.SelectMany(v => v).ToArray());
            var features = WeSpeakerAudio.LogMelFilterbank(samples, 16000, CancellationToken.None);
            Save(name, "features", feature.GetString() ?? "", features.ToArray());
            var encoded = Execute(encoding, "fbank_features", features, "/resnet/pool/Reshape_output_0");
            Save(name, "encoded", window.GetProperty("encoded").GetString() ?? "", encoded.ToArray());
            var pooled = new List<float>();
            foreach (var mask in masks)
            {
                int frames = encoded.Dimensions[2]; var resized = Enumerable.Range(0, frames).Select(t => mask[t * mask.Length / frames]).ToArray();
                pooled.AddRange(((Tensor<float>)Call("WeSpeakerPooling", "PoolPipeline", null, encoded, resized, CancellationToken.None)).ToArray());
            }
            Save(name, "pooled", window.GetProperty("pooled").GetString() ?? "", pooled.ToArray());
            var vectors = (WeSpeakerEmbedding[])Call("WeSpeakerEmbedder", "ExtractPipeline", embedder, samples, masks, CancellationToken.None);
            Array.Copy(vectors, 0, embeddings, i * 3, 3); Save(name, "vectors", window.GetProperty("vectors").GetString() ?? "", vectors.SelectMany(v => v.Values).ToArray());
        }
        var count = (int[])Call("Community1Timeline", "Count", null, activity, chunks, CancellationToken.None);
        Save(name, "count", c.GetProperty("count").GetString() ?? "", count.Select(v => (float)v).ToArray());
        if (c.TryGetProperty("labels", out var labels))
        {
            var result = clusterer.Cluster(embeddings, new DenseTensor<bool>(activity, new[] { chunks, 589, 3 }), CancellationToken.None);
            Save(name, "labels", labels.GetString() ?? "", result.Labels.Select(v => (float)v).ToArray());
            // Centroids are saved as double by the public API replay; this trace preserves its float model boundaries.
        }
        Console.WriteLine(name + " complete");
    }
}
finally { segmentation.Reset(); encoding.Reset(); }
File.WriteAllText(Path.Combine(output, "result.json"), JsonSerializer.Serialize(new { execution_complete = true, reference_sha256 = evidence.ManifestSha, assemblies = Evidence.Assemblies(), reports }, new JsonSerializerOptions { WriteIndented = true }));
