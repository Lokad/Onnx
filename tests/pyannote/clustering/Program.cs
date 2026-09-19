using System.Text.Json;
using Lokad.Onnx;
using System.Security.Cryptography;
using System.Reflection;
using System.Text;
using System.Diagnostics;

string source = Path.GetFullPath(args[0]), destination = Path.GetFullPath(args[1]);
if (File.Exists(destination)) throw new IOException("Output exists.");
using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(source, "reference.json"))); var root = doc.RootElement;
string Sha(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
string FileSha(string path) => Sha(File.ReadAllBytes(path));
string Resource(string name)
{
    using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(name) ?? throw new InvalidDataException(name);
    using var reader = new StreamReader(stream); return reader.ReadToEnd().Replace("\r\n", "\n", StringComparison.Ordinal);
}
using var pinDoc = JsonDocument.Parse(Resource("clustering-pins.json")); var pins = pinDoc.RootElement;
foreach (string key in new[] { "source_revision", "source_hashes", "numpy", "scipy" })
    if (!JsonElement.DeepEquals(root.GetProperty(key), pins.GetProperty(key))) throw new InvalidDataException("Reference identity: " + key);
if (root.GetProperty("generator_lf_sha256").GetString() != Sha(Encoding.UTF8.GetBytes(Resource("clustering-generator.py")))
    || root.GetProperty("prepare_lf_sha256").GetString() != Sha(Encoding.UTF8.GetBytes(Resource("clustering-prepare.py")))
    || root.GetProperty("prepared_sha256").GetString() != FileSha(Path.Combine(source, "prepared.json"))) throw new InvalidDataException("Reference source/model digest.");
foreach (string group in new[] { "cases", "hierarchy" })
{
    var expected = pins.GetProperty(group).EnumerateArray().Select(v => v.GetString()).ToArray();
    var actual = root.GetProperty(group).EnumerateArray().Select(v => v.GetProperty("name").GetString()).ToArray();
    if (!actual.SequenceEqual(expected)) throw new InvalidDataException("Reference coverage mismatch.");
}
var parameters = new Community1Parameters(Path.Combine(source, "prepared.json"));
var publicApi = new Community1Clusterer(Path.Combine(source, "prepared.json"));
double[] D(JsonElement e)
{
    string dtype = e.GetProperty("dtype").GetString() ?? "";
    if (!new[] { "float32", "float64", "int8", "int32", "int64" }.Contains(dtype)) throw new InvalidDataException("Reference dtype.");
    long length = 1; foreach (var d in e.GetProperty("shape").EnumerateArray()) { int size = d.GetInt32(); if (size < 0) throw new InvalidDataException("Negative shape."); length = checked(length * size); }
    var values = e.GetProperty("values").EnumerateArray().Select(v => v.GetDouble()).ToArray();
    if (length != values.Length || values.Any(v => !double.IsFinite(v)) || dtype.StartsWith("int", StringComparison.Ordinal) && values.Any(v => v != Math.Truncate(v)))
        throw new InvalidDataException("Reference shape or values.");
    return values;
}
float[] F(JsonElement e)
{
    if (e.GetProperty("dtype").GetString() != "float32") throw new InvalidDataException("Reference float input dtype.");
    return D(e).Select(v => (float)v).ToArray();
}
int[] Shape(JsonElement e) => e.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
var reports = new List<object>(); bool passed = true; double maximum = 0; int values = 0;
var partitions = new List<object>(); var publicResults = new List<object>();
void Compare(string name, string stage, double[] actual, double[] expected)
{
    if (actual.Length != expected.Length) throw new InvalidDataException(name + " " + stage + " length");
    double error = 0; int bad = 0;
    for (int i = 0; i < actual.Length; i++) { double e = Math.Abs(actual[i] - expected[i]) / Math.Max(1, Math.Abs(expected[i])); if (!double.IsFinite(e)) throw new InvalidDataException("Nonfinite"); error = Math.Max(error, e); if (e > 1e-4) bad++; }
    reports.Add(new { name, stage, error, bad, actual }); maximum = Math.Max(maximum, error); passed &= bad == 0; values += actual.Length;
    if (bad != 0) Console.WriteLine($"FAIL {name} {stage} error={error:R} bad={bad}");
}
int[] Mapping(int[] labels, double[] expected)
{
    var mapping = Enumerable.Repeat(-1, labels.Max() + 1).ToArray();
    for (int i = 0; i < labels.Length; i++) { int l = labels[i], e = (int)expected[i]; if (mapping[l] >= 0 && mapping[l] != e) throw new InvalidDataException("Partition disagreement"); mapping[l] = e; }
    if (mapping.Distinct().Count() != mapping.Length) throw new InvalidDataException("Partition collapse"); return mapping;
}
foreach (var c in root.GetProperty("hierarchy").EnumerateArray())
{
    string name = c.GetProperty("name").GetString() ?? ""; var points = c.GetProperty("points"); var shape = Shape(points);
    var h = Community1Math.Hierarchy(F(points), shape[0], shape[1], c.GetProperty("threshold").GetDouble(), CancellationToken.None);
    var map = Mapping(h.Labels, D(c.GetProperty("labels")));
    partitions.Add(new { name, labels = h.Labels, mapping = map });
    Compare(name, "merge-distances", h.Distances, D(c.GetProperty("hierarchy")).Where((_, i) => i % 4 == 2).ToArray());
}
foreach (var c in root.GetProperty("cases").EnumerateArray())
{
    string name = c.GetProperty("name").GetString() ?? ""; var embeddings = F(c.GetProperty("embeddings")); var shape = Shape(c.GetProperty("embeddings"));
    var activity = F(c.GetProperty("activity")); int chunks = shape[0], speakers = shape[1], frames = Shape(c.GetProperty("activity"))[1];
    var training = new List<int>(); var active = new bool[chunks * speakers];
    for (int ch = 0; ch < chunks; ch++) for (int sp = 0; sp < speakers; sp++)
    {
        int clean = 0;
        for (int t = 0; t < frames; t++)
        {
            float value = activity[(ch * frames + t) * speakers + sp]; active[ch * speakers + sp] |= value > 0;
            float total = 0; for (int s = 0; s < speakers; s++) total += activity[(ch * frames + t) * speakers + s];
            if (total == 1 && value == 1) clean++;
        }
        if (clean * 5 >= frames) training.Add(ch * speakers + sp);
    }
    if (!training.SequenceEqual(D(c.GetProperty("training_indices")).Select(v => (int)v))) throw new InvalidDataException("Training selection");
    float[] train = training.SelectMany(i => embeddings.Skip(i * 256).Take(256)).ToArray(); int count = training.Count;
    double[] centers; int clusters; int[] nativeOrder;
    if (count == 1) { centers = train.Select(v => (double)v).ToArray(); clusters = 1; nativeOrder = new[] { 0 }; }
    else
    {
        var normalized = Community1Math.Normalize(train, count, 256); Compare(name, "normalized", normalized.Select(v => (double)v).ToArray(), D(c.GetProperty("normalized")));
        var h = Community1Math.Hierarchy(normalized, count, 256, .6, CancellationToken.None); var map = Mapping(h.Labels, D(c.GetProperty("labels")));
        partitions.Add(new { name, labels = h.Labels, mapping = map });
        var transformed = parameters.Transform(train, count, CancellationToken.None); Compare(name, "transformed", transformed, D(c.GetProperty("transformed")));
        var state = Community1Math.Refine(transformed, count, 128, parameters.Phi, h.Labels, CancellationToken.None); int k = map.Length;
        double[] Reorder(double[] a, int rows, int cols, bool column)
        {
            var result = new double[a.Length];
            for (int r = 0; r < rows; r++) for (int col = 0; col < cols; col++) result[(column ? r : map[r]) * cols + (column ? map[col] : col)] = a[r * cols + col]; return result;
        }
        Compare(name, "responsibilities", Reorder(state.Responsibilities, count, k, true), D(c.GetProperty("responsibilities")));
        Compare(name, "priors", Reorder(state.Priors, k, 1, false), D(c.GetProperty("priors")));
        Compare(name, "objective", state.Objective, D(c.GetProperty("objective")));
        Compare(name, "alpha", Reorder(state.Alpha, k, 128, false), D(c.GetProperty("alpha")));
        Compare(name, "inverse-precision", Reorder(state.InversePrecision, k, 128, false), D(c.GetProperty("inverse_precision")));
        var kept = Enumerable.Range(0, k).Where(s => state.Priors[s] > 1e-7).ToArray(); clusters = kept.Length;
        var nativeKept = Enumerable.Range(0, k).Where(s => D(c.GetProperty("priors"))[s] > 1e-7).ToArray();
        nativeOrder = kept.Select(s => Array.IndexOf(nativeKept, map[s])).ToArray(); centers = new double[clusters * 256];
        for (int s = 0; s < clusters; s++)
        {
            double weight = 0; for (int i = 0; i < count; i++) weight += state.Responsibilities[i * k + kept[s]];
            for (int d = 0; d < 256; d++) { double sum = 0; for (int i = 0; i < count; i++) sum += state.Responsibilities[i * k + kept[s]] * train[i * 256 + d]; centers[s * 256 + d] = sum / weight; }
        }
    }
    var nativeCenters = new double[centers.Length]; for (int s = 0; s < clusters; s++) Array.Copy(centers, s * 256, nativeCenters, nativeOrder[s] * 256, 256);
    Compare(name, "centroids", nativeCenters, D(c.GetProperty("centroids")));
    var soft = new double[chunks * speakers * clusters];
    if (count == 1) Array.Fill(soft, 1);
    else
    {
        for (int i = 0; i < chunks * speakers; i++) for (int s = 0; s < clusters; s++)
        {
            double dot = 0, nx = 0, ny = 0;
            for (int d = 0; d < 256; d++) { double x = embeddings[i * 256 + d], y = centers[s * 256 + d]; dot += x * y; nx += x * x; ny += y * y; }
            double distance = Math.Clamp(1 - dot / Math.Sqrt(nx * ny), 0, 2); soft[i * clusters + s] = 2 - distance;
        }
        double inactive = soft.Min() - 1;
        for (int i = 0; i < active.Length; i++) if (!active[i]) for (int s = 0; s < clusters; s++) soft[i * clusters + s] = inactive;
    }
    var nativeSoft = new double[soft.Length]; for (int i = 0; i < chunks * speakers; i++) for (int s = 0; s < clusters; s++) nativeSoft[i * clusters + nativeOrder[s]] = soft[i * clusters + s];
    Compare(name, "soft", nativeSoft, D(c.GetProperty("soft")));
    var hard = new int[chunks * speakers];
    for (int ch = 0; ch < chunks; ch++)
    {
        var h = count == 1 ? new int[speakers] : Community1Math.Match(soft.Skip(ch * speakers * clusters).Take(speakers * clusters).ToArray(), speakers, clusters);
        for (int s = 0; s < speakers; s++) hard[ch * speakers + s] = h[s] < 0 ? h[s] : nativeOrder[h[s]];
    }
    Compare(name, "hard", hard.Select(v => (double)v).ToArray(), D(c.GetProperty("hard")));
    var entries = Enumerable.Range(0, chunks * speakers).Select(i => new WeSpeakerEmbedding(
        Array.AsReadOnly(embeddings.Skip(i * 256).Take(256).ToArray()), WeSpeakerEmbeddingStatus.Completed, 13, 13)).ToArray();
    var binaryActivity = new DenseTensor<bool>(activity.Select(v => v == 1).ToArray(), new[] { chunks, frames, speakers });
    var inputBits = embeddings.Select(BitConverter.SingleToInt32Bits).ToArray();
    var expectedLabels = D(c.GetProperty("hard")).Select(v => (int)v).ToArray();
    for (int i = 0; i < expectedLabels.Length; i++) if (!active[i]) expectedLabels[i] = -2;
    var canonical = Enumerable.Repeat(-1, clusters).ToArray(); int nextLabel = 0;
    for (int i = 0; i < expectedLabels.Length; i++) if (expectedLabels[i] >= 0) { int old = expectedLabels[i]; if (canonical[old] < 0) canonical[old] = nextLabel++; expectedLabels[i] = canonical[old]; }
    for (int i = 0; i < clusters; i++) if (canonical[i] < 0) canonical[i] = nextLabel++;
    var expectedCentroids = new double[clusters * 256]; var nativeCentroids = D(c.GetProperty("centroids"));
    for (int i = 0; i < clusters; i++) Array.Copy(nativeCentroids, i * 256, expectedCentroids, canonical[i] * 256, 256);
    var held = publicApi.Cluster(entries, binaryActivity, CancellationToken.None);
    for (int repeat = 0; repeat < 2; repeat++)
    {
        var result = repeat == 0 ? held : publicApi.Cluster(entries, binaryActivity, CancellationToken.None);
        if (result.TrainingEmbeddings != count || result.Chunks != chunks || result.LocalSpeakers != speakers
            || !result.Labels.SequenceEqual(expectedLabels)) throw new InvalidDataException("Public clustering contract.");
        Compare(name, "public-centroids-" + repeat, result.Centroids.SelectMany(v => v).ToArray(), expectedCentroids);
        if (!result.Labels.SequenceEqual(held.Labels) || !result.Centroids.SelectMany(v => v).SequenceEqual(held.Centroids.SelectMany(v => v))) throw new InvalidDataException("Public repeat changed.");
        publicResults.Add(new { name, repeat, labels = result.Labels.ToArray(), clusters = result.Centroids.Count, training = result.TrainingEmbeddings });
    }
    try { publicApi.Cluster(entries, binaryActivity, new CancellationToken(true)); throw new InvalidOperationException("Cancellation accepted."); }
    catch (OperationCanceledException) { }
    if (!publicApi.Cluster(entries, binaryActivity, CancellationToken.None).Labels.SequenceEqual(held.Labels)
        || !inputBits.SequenceEqual(embeddings.Select(BitConverter.SingleToInt32Bits))
        || !activity.Select(v => v == 1).SequenceEqual(binaryActivity.ToArray())) throw new InvalidDataException("Request recovery/ownership.");
    Console.WriteLine(name + " complete");
}
using var process = Process.GetCurrentProcess();
foreach (ProcessModule module in process.Modules) if (module.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)) throw new InvalidDataException("Native ORT loaded.");
File.WriteAllText(destination, JsonSerializer.Serialize(new { passed, values, maximum, reports, partitions, publicResults,
    reference_sha256 = FileSha(Path.Combine(source, "reference.json")), prepared_sha256 = FileSha(Path.Combine(source, "prepared.json")),
    core_sha256 = FileSha(typeof(Tensor<float>).Assembly.Location), data_sha256 = FileSha(typeof(Community1Clusterer).Assembly.Location),
    runner_sha256 = FileSha(Assembly.GetExecutingAssembly().Location), runtime = Environment.Version.ToString(), peak = process.PeakWorkingSet64 }, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"Passed={passed} values={values} max={maximum:R}");return passed ? 0 : 1;
