namespace Lokad.Onnx;

// The learned transform and VBx equations follow pyannote.audio / BUTSpeechFIT VBx.
// Adapted to managed request-local storage; see NOTICE.txt and Apache-2.0.txt.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;

internal sealed class Community1Parameters
{
    readonly double[] mean1, mean2, lda, mean, transform;
    internal readonly double[] Phi;
    internal Community1Parameters(string path)
    {
        if (new FileInfo(path).Length > 4 * 1024 * 1024) throw new InvalidDataException("Prepared model exceeds 4 MiB.");
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path)); var root = doc.RootElement;
        if (root.GetProperty("schema").GetInt32() != 1 || root.GetProperty("input_dimensions").GetInt32() != 256
            || root.GetProperty("output_dimensions").GetInt32() != 128) throw new InvalidDataException("Unsupported prepared model.");
        mean1 = Vector(root.GetProperty("mean1"), 256); mean2 = Vector(root.GetProperty("mean2"), 128);
        mean = Vector(root.GetProperty("mean"), 128); Phi = Vector(root.GetProperty("phi"), 128);
        if (Phi.Any(v => v <= 0)) throw new InvalidDataException("Covariances must be positive.");
        lda = Matrix(root.GetProperty("lda"), 256, 128); transform = Matrix(root.GetProperty("transform"), 128, 128);
    }
    static double[] Vector(JsonElement element, int size)
    {
        if (element.GetArrayLength() != size) throw new InvalidDataException("Prepared vector dimension.");
        var a = element.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        if (a.Any(v => !double.IsFinite(v))) throw new InvalidDataException("Prepared parameters must be finite.");
        return a;
    }
    static double[] Matrix(JsonElement element, int rows, int columns)
    {
        if (element.GetArrayLength() != rows) throw new InvalidDataException("Prepared matrix dimension.");
        return element.EnumerateArray().SelectMany(x => Vector(x, columns)).ToArray();
    }
    internal double[] Transform(float[] vectors, int count, CancellationToken cancellation)
    {
        if (vectors.Length != checked(count * 256)) throw new ArgumentException("Embedding dimension.");
        var output = new double[count * 128]; var centered = new double[256]; var projected = new double[128];
        for (int n = 0; n < count; n++)
        {
            cancellation.ThrowIfCancellationRequested(); double norm = 0;
            for (int d = 0; d < 256; d++) { double x = vectors[n * 256 + d]; if (!double.IsFinite(x)) throw new ArgumentException("Nonfinite embedding."); centered[d] = x - mean1[d]; norm += centered[d] * centered[d]; }
            if (!(norm > 0) || !double.IsFinite(norm)) throw new ArgumentException("Invalid centered embedding norm.");
            norm = Math.Sqrt(norm);
            for (int d = 0; d < 256; d++) centered[d] = 16 * (centered[d] / norm);
            norm = 0;
            for (int d = 0; d < 128; d++)
            {
                double sum = 0; for (int k = 0; k < 256; k++) sum += lda[k * 128 + d] * centered[k];
                projected[d] = sum - mean2[d]; norm += projected[d] * projected[d];
            }
            if (!(norm > 0) || !double.IsFinite(norm)) throw new ArgumentException("Invalid projected embedding norm.");
            norm = Math.Sqrt(norm);
            for (int d = 0; d < 128; d++) projected[d] = Math.Sqrt(128) * (projected[d] / norm) - mean[d];
            for (int d = 0; d < 128; d++)
            {
                double sum = 0; for (int k = 0; k < 128; k++) sum += projected[k] * transform[d * 128 + k];
                if (!double.IsFinite(sum)) throw new InvalidDataException("Nonfinite transformed embedding.");
                output[n * 128 + d] = sum;
            }
        }
        return output;
    }
}

internal sealed record CentroidHierarchy(int[] Labels, double[] Distances, int[] Left, int[] Right);
internal sealed record VbxState(double[] Responsibilities, double[] Priors, double[] Objective, double[] Alpha, double[] InversePrecision);

internal static class Community1Math
{
    internal static float[] Normalize(float[] values, int count, int dimensions)
    {
        if (values.Length != checked(count * dimensions)) throw new ArgumentException("Vector dimensions.");
        var output = new float[values.Length];
        for (int i = 0; i < count; i++)
        {
            double sum = 0;
            for (int j = 0; j < dimensions; j++) { float x = values[i * dimensions + j]; if (!float.IsFinite(x)) throw new ArgumentException("Nonfinite embedding."); sum += (float)(x * x); }
            float norm = MathF.Sqrt((float)sum);
            if (!(norm > 0) || !float.IsFinite(norm)) throw new ArgumentException("Invalid embedding norm.");
            for (int j = 0; j < dimensions; j++) output[i * dimensions + j] = values[i * dimensions + j] / norm;
        }
        return output;
    }

    internal static CentroidHierarchy Hierarchy(float[] points, int count, int dimensions, double threshold, CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();
        if (count < 1 || count > 4096 || dimensions < 1 || points.Length != checked(count * dimensions) || !double.IsFinite(threshold) || threshold < 0)
            throw new ArgumentException("Hierarchy dimensions or threshold.");
        if (points.Any(v => !float.IsFinite(v))) throw new ArgumentException("Nonfinite hierarchy input.");
        var size = Enumerable.Repeat(1, count).ToArray(); var ids = Enumerable.Range(0, count).ToArray();
        var active = Enumerable.Repeat(true, count).ToArray(); var pairDistances = new double[count * count];
        var queue = new PriorityQueue<(int A, int B, int IdA, int IdB), (double Distance, int IdA, int IdB)>();
        for (int i = 0; i < count; i++)
        {
            cancellation.ThrowIfCancellationRequested();
            for (int j = i + 1; j < count; j++)
            {
                double sum = 0; for (int d = 0; d < dimensions; d++) { double x = (double)points[i * dimensions + d] - points[j * dimensions + d]; sum += x * x; }
                pairDistances[i * count + j] = pairDistances[j * count + i] = sum;
                queue.Enqueue((i, j, i, j), (sum, i, j));
            }
        }
        var heights = new double[count - 1]; var left = new int[count - 1]; var right = new int[count - 1]; var maxima = new double[2 * count - 1];
        for (int step = 0; step < count - 1; step++)
        {
            cancellation.ThrowIfCancellationRequested();
            (int A, int B, int IdA, int IdB) pair;
            do { pair = queue.Dequeue(); } while (!active[pair.A] || !active[pair.B] || ids[pair.A] != pair.IdA || ids[pair.B] != pair.IdB);
            int a = pair.A, b = pair.B, node = count + step, na = size[a], nb = size[b];
            double distance = pairDistances[a * count + b]; heights[step] = Math.Sqrt(Math.Max(0, distance));
            left[step] = Math.Min(ids[a], ids[b]); right[step] = Math.Max(ids[a], ids[b]);
            maxima[node] = Math.Max(heights[step], Math.Max(maxima[ids[a]], maxima[ids[b]]));
            for (int j = 0; j < count; j++) if (active[j] && j != a && j != b)
            {
                double value = (na * pairDistances[a * count + j] + nb * pairDistances[b * count + j]) / (na + nb)
                    - (double)na * nb / ((double)(na + nb) * (na + nb)) * distance;
                pairDistances[a * count + j] = pairDistances[j * count + a] = Math.Max(0, value);
            }
            active[b] = false; size[a] += nb; ids[a] = node;
            for (int j = 0; j < count; j++) if (active[j] && j != a)
            {
                int i0 = ids[a] < ids[j] ? a : j, i1 = i0 == a ? j : a;
                queue.Enqueue((i0, i1, ids[i0], ids[i1]), (pairDistances[a * count + j], ids[i0], ids[i1]));
            }
        }
        var labels = Enumerable.Repeat(-1, count).ToArray(); int next = 0;
        var pending = new Stack<(int Node, int Label)>(); pending.Push((2 * count - 2, -1));
        while (pending.TryPop(out var item))
        {
            cancellation.ThrowIfCancellationRequested();
            int node = item.Node, label = item.Label;
            if (label < 0 && (node < count || maxima[node] <= threshold)) label = next++;
            if (node < count) labels[node] = label;
            else { pending.Push((right[node - count], label)); pending.Push((left[node - count], label)); }
        }
        // Labels are stable by the first input occurrence, independent of tree traversal order.
        var canonical = new Dictionary<int, int>();
        for (int i = 0; i < count; i++) { if (!canonical.TryGetValue(labels[i], out int label)) { label = canonical.Count; canonical.Add(labels[i], label); } labels[i] = label; }
        return new CentroidHierarchy(labels, heights, left, right);
    }

    internal static VbxState Refine(double[] x, int count, int dimensions, double[] phi, int[] labels, CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();
        if (count < 1 || count > 4096 || dimensions < 1 || phi.Length != dimensions || labels.Length != count || x.Length != checked(count * dimensions)
            || labels.Any(v => v < 0 || v >= count) || x.Any(v => !double.IsFinite(v)) || phi.Any(v => !double.IsFinite(v) || v <= 0))
            throw new ArgumentException("VBx input contract.");
        int clusters = labels.Max() + 1;
        if (labels.Distinct().Count() != clusters) throw new ArgumentException("Cluster labels must be contiguous.");
        const double fa = .07, fb = .8;
        var gamma = new double[count * clusters]; var priors = Enumerable.Repeat(1.0 / clusters, clusters).ToArray();
        double smooth = Math.Exp(7), normalizer = smooth + clusters - 1;
        for (int i = 0; i < count; i++) for (int s = 0; s < clusters; s++) gamma[i * clusters + s] = (s == labels[i] ? smooth : 1) / normalizer;
        var rho = new double[x.Length]; var constants = new double[count];
        for (int i = 0; i < count; i++)
        {
            double sum = 0; for (int d = 0; d < dimensions; d++) { double v = x[i * dimensions + d]; sum += v * v; rho[i * dimensions + d] = v * Math.Sqrt(phi[d]); }
            constants[i] = -.5 * (sum + dimensions * Math.Log(2 * Math.PI));
        }
        var alpha = new double[clusters * dimensions]; var inv = new double[alpha.Length]; var logP = new double[clusters]; var history = new List<double>();
        for (int iteration = 0; iteration < 20; iteration++)
        {
            cancellation.ThrowIfCancellationRequested();
            for (int s = 0; s < clusters; s++)
            {
                cancellation.ThrowIfCancellationRequested();
                double sum = 0; for (int i = 0; i < count; i++) sum += gamma[i * clusters + s];
                for (int d = 0; d < dimensions; d++)
                {
                    int k = s * dimensions + d; inv[k] = 1 / (1 + fa / fb * sum * phi[d]);
                    double dot = 0; for (int i = 0; i < count; i++) dot += gamma[i * clusters + s] * rho[i * dimensions + d];
                    alpha[k] = fa / fb * inv[k] * dot;
                }
            }
            var correction = new double[clusters];
            for (int s = 0; s < clusters; s++) for (int d = 0; d < dimensions; d++) { int k = s * dimensions + d; correction[s] += (inv[k] + alpha[k] * alpha[k]) * phi[d]; }
            double likelihood = 0;
            for (int i = 0; i < count; i++)
            {
                cancellation.ThrowIfCancellationRequested();
                double max = double.NegativeInfinity;
                for (int s = 0; s < clusters; s++)
                {
                    double dot = 0; for (int d = 0; d < dimensions; d++) dot += rho[i * dimensions + d] * alpha[s * dimensions + d];
                    logP[s] = fa * (dot - .5 * correction[s] + constants[i]) + Math.Log(priors[s] + 1e-8); max = Math.Max(max, logP[s]);
                }
                double total = 0; for (int s = 0; s < clusters; s++) total += Math.Exp(logP[s] - max);
                double logSum = max + Math.Log(total); likelihood += logSum;
                for (int s = 0; s < clusters; s++) gamma[i * clusters + s] = Math.Exp(logP[s] - logSum);
            }
            double norm = 0;
            for (int s = 0; s < clusters; s++) { double sum = 0; for (int i = 0; i < count; i++) sum += gamma[i * clusters + s]; priors[s] = sum; norm += sum; }
            for (int s = 0; s < clusters; s++) priors[s] /= norm;
            double penalty = 0; for (int k = 0; k < inv.Length; k++) penalty += Math.Log(inv[k]) - inv[k] - alpha[k] * alpha[k] + 1;
            double objective = likelihood + fb * .5 * penalty;
            if (!double.IsFinite(objective)) throw new InvalidDataException("Nonfinite VBx objective.");
            history.Add(objective);
            if (iteration > 0 && objective - history[iteration - 1] < 1e-4) break;
        }
        return new VbxState(gamma, priors, history.ToArray(), alpha, inv);
    }

    internal static int[] Match(double[] scores, int rows, int columns)
    {
        if (rows < 1 || columns < 1 || scores.Length != checked(rows * columns) || scores.Any(v => !double.IsFinite(v))) throw new ArgumentException("Assignment input.");
        bool transpose = rows > columns; int n = Math.Min(rows, columns), m = Math.Max(rows, columns);
        var u = new double[n + 1]; var v = new double[m + 1]; var p = new int[m + 1]; var way = new int[m + 1];
        for (int i = 1; i <= n; i++)
        {
            p[0] = i; int j0 = 0; var min = Enumerable.Repeat(double.PositiveInfinity, m + 1).ToArray(); var used = new bool[m + 1];
            do
            {
                used[j0] = true; int i0 = p[j0], j1 = 0; double delta = double.PositiveInfinity;
                for (int j = 1; j <= m; j++) if (!used[j])
                {
                    double score = transpose ? scores[(j - 1) * columns + i0 - 1] : scores[(i0 - 1) * columns + j - 1];
                    double cur = -score - u[i0] - v[j];
                    if (cur < min[j]) { min[j] = cur; way[j] = j0; }
                    if (min[j] < delta) { delta = min[j]; j1 = j; }
                }
                for (int j = 0; j <= m; j++) if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else min[j] -= delta;
                j0 = j1;
            } while (p[j0] != 0);
            do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0 != 0);
        }
        var result = Enumerable.Repeat(-2, rows).ToArray();
        for (int j = 1; j <= m; j++) if (p[j] != 0) { if (transpose) result[j - 1] = p[j] - 1; else result[p[j] - 1] = j - 1; }
        return result;
    }
}
