namespace Lokad.Onnx.Backend.Tests;

using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Text.Json;

public class Community1ClusteringTests
{
    static object Call(string method, params object[] args)
    {
        var type = typeof(Community1Clusterer).Assembly.GetType("Lokad.Onnx.Community1Math", true) ?? throw new InvalidOperationException();
        try { return type.GetMethod(method, BindingFlags.Static | BindingFlags.NonPublic)?.Invoke(null, args) ?? throw new InvalidOperationException(); }
        catch (TargetInvocationException e) when (e.InnerException is not null) { ExceptionDispatchInfo.Capture(e.InnerException).Throw(); throw; }
    }
    static T Property<T>(object target, string name) => (T)(target.GetType().GetProperty(name)?.GetValue(target) ?? throw new InvalidOperationException());
    static int[] Hierarchy(float[] points, int count, int dimensions, double threshold) => Property<int[]>(Call("Hierarchy", points, count, dimensions, threshold, CancellationToken.None), "Labels");

    [Fact]
    public void CentroidInversionCannotMergeAcrossAnEarlierLargerDistance()
    {
        // First merge is about .9958; the resulting centroid is only .864 away from the last point.
        // A .9 cut still has three clusters: all ancestor merge heights must qualify.
        var points = new float[] { 0, 0, 1, 0, .5f, .86f };
        Assert.Equal(new[] { 0, 1, 2 }, Hierarchy(points, 3, 2, .9));
        Assert.Equal(new[] { 0, 0, 0 }, Hierarchy(points, 3, 2, 1.01));
    }

    [Fact]
    public void DuplicateDistancesAndThresholdBoundaryProduceStablePartitions()
    {
        var points = new float[] { 0, 0, 1, 1 };
        Assert.Equal(new[] { 0, 0, 1, 1 }, Hierarchy(points, 4, 1, 0));
        Assert.Equal(new[] { 0, 0, 1, 1 }, Hierarchy(points, 4, 1, Math.BitDecrement(1)));
        Assert.Equal(new[] { 0, 0, 0, 0 }, Hierarchy(points, 4, 1, 1));
        Assert.Equal(new[] { 0 }, Hierarchy(new float[] { 3 }, 1, 1, 0));
        Assert.Equal(new float[] { 0, 0, 1, 1 }, points);
    }

    [Theory]
    [InlineData(1, 3)] [InlineData(3, 1)] [InlineData(2, 3)] [InlineData(3, 2)] [InlineData(3, 3)] [InlineData(4, 5)]
    public void AssignmentMatchesExhaustiveGlobalOptimum(int rows, int columns)
    {
        var random = new Random(731 + rows * columns);
        for (int repeat = 0; repeat < 30; repeat++)
        {
            double[] scores = Enumerable.Range(0, rows * columns).Select(_ => (double)random.Next(-3, 4)).ToArray();
            var before = scores.ToArray(); var actual = (int[])Call("Match", scores, rows, columns);
            Assert.Equal(Math.Min(rows, columns), actual.Count(v => v >= 0)); Assert.Equal(actual.Where(v => v >= 0).Distinct().Count(), actual.Count(v => v >= 0));
            double measured = actual.Select((s, i) => s < 0 ? 0 : scores[i * columns + s]).Sum();
            double best = double.NegativeInfinity;
            void Search(int row, int used, int assigned, double total)
            {
                if (row == rows) { if (assigned == Math.Min(rows, columns)) best = Math.Max(best, total); return; }
                if (rows > columns) Search(row + 1, used, assigned, total);
                for (int s = 0; s < columns; s++) if ((used & (1 << s)) == 0) Search(row + 1, used | (1 << s), assigned + 1, total + scores[row * columns + s]);
            }
            Search(0, 0, 0, 0); Assert.Equal(best, measured); Assert.Equal(before, scores);
        }
    }

    [Fact]
    public void OneClusterVbxMatchesClosedFormPosteriorAndStopsAfterTwoIterations()
    {
        double[] x = { 1, 2, 3, 4, 5, 6 }, phi = { 1, 4 };
        var result = Call("Refine", x, 3, 2, phi, new[] { 0, 0, 0 }, CancellationToken.None);
        Assert.Equal(new double[] { 1, 1, 1 }, Property<double[]>(result, "Responsibilities"));
        Assert.Equal(new double[] { 1 }, Property<double[]>(result, "Priors"));
        Assert.Equal(2, Property<double[]>(result, "Objective").Length);
        double ratio = .07 / .8;
        var inv = Property<double[]>(result, "InversePrecision"); var alpha = Property<double[]>(result, "Alpha");
        Assert.Equal(1 / (1 + ratio * 3), inv[0], 12); Assert.Equal(1 / (1 + ratio * 3 * 4), inv[1], 12);
        Assert.Equal(ratio * inv[0] * 9, alpha[0], 12); Assert.Equal(ratio * inv[1] * 24, alpha[1], 12);
        Assert.Equal(new double[] { 1, 2, 3, 4, 5, 6 }, x); Assert.Equal(new double[] { 1, 4 }, phi);
    }

    [Fact]
    public void NumericalPrimitivesRejectMalformedAndCancelledRequests()
    {
        Assert.Throws<ArgumentException>(() => Call("Normalize", new float[] { 0, 0 }, 1, 2));
        Assert.Throws<ArgumentException>(() => Hierarchy(new float[] { float.NaN }, 1, 1, .6));
        Assert.Throws<ArgumentException>(() => Hierarchy(new float[] { 1 }, 1, 1, -1));
        Assert.Throws<ArgumentException>(() => Call("Refine", new double[] { 1, 2 }, 2, 1, new double[] { 1 }, new[] { 0, 2 }, CancellationToken.None));
        Assert.Throws<ArgumentException>(() => Call("Refine", new double[] { 1 }, 1, 1, new double[] { -1 }, new[] { 0 }, CancellationToken.None));
        Assert.Throws<ArgumentException>(() => Call("Match", new double[] { double.NaN }, 1, 1));
        Assert.Throws<OperationCanceledException>(() => Call("Hierarchy", new float[] { 1, 2 }, 2, 1, .6, new CancellationToken(true)));
        Assert.Throws<OperationCanceledException>(() => Call("Refine", new double[] { 1 }, 1, 1, new double[] { 1 }, new[] { 0 }, new CancellationToken(true)));
    }

    sealed class Model : IDisposable
    {
        internal readonly string Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "community1-" + Guid.NewGuid().ToString("N") + ".json");
        internal Model()
        {
            File.WriteAllText(Path, JsonSerializer.Serialize(new { schema = 1, input_dimensions = 256, output_dimensions = 128,
                mean1 = new double[256], mean2 = new double[128], mean = new double[128], phi = Enumerable.Repeat(1d, 128).ToArray(),
                lda = Enumerable.Range(0, 256).Select(i => Enumerable.Range(0, 128).Select(j => i == j ? 1d : 0).ToArray()).ToArray(),
                transform = Enumerable.Range(0, 128).Select(i => Enumerable.Range(0, 128).Select(j => i == j ? 1d : 0).ToArray()).ToArray() }));
        }
        public void Dispose() => File.Delete(Path);
    }
    static WeSpeakerEmbedding Vector(int axis)
    {
        var a = new float[256]; a[axis] = 1;
        return new WeSpeakerEmbedding(Array.AsReadOnly(a), WeSpeakerEmbeddingStatus.Completed, 13, 13);
    }
    static WeSpeakerEmbedding Missing() => new(Array.AsReadOnly(Array.Empty<float>()), WeSpeakerEmbeddingStatus.InsufficientFrames, 13, 0);
    static DenseTensor<bool> Mask(int chunks, int frames, int speakers, Func<int, int, int, bool> value)
    {
        var a = new bool[chunks * frames * speakers];
        for (int c = 0; c < chunks; c++) for (int t = 0; t < frames; t++) for (int s = 0; s < speakers; s++) a[(c * frames + t) * speakers + s] = value(c, t, s);
        return new DenseTensor<bool>(a, new[] { chunks, frames, speakers });
    }

    [Fact]
    public void LearnedTransformMatchesIndependentIdentityProjection()
    {
        using var model = new Model();
        var type = typeof(Community1Clusterer).Assembly.GetType("Lokad.Onnx.Community1Parameters", true) ?? throw new InvalidOperationException();
        var parameters = Activator.CreateInstance(type, BindingFlags.Instance | BindingFlags.NonPublic, null, new object[] { model.Path }, null) ?? throw new InvalidOperationException();
        var vector = new float[256]; vector[0] = 3; vector[1] = 4; vector[255] = 12;
        var actual = (double[])(type.GetMethod("Transform", BindingFlags.Instance | BindingFlags.NonPublic)?.Invoke(parameters, new object[] { vector, 1, CancellationToken.None }) ?? throw new InvalidOperationException());
        Assert.Equal(Math.Sqrt(128) * .6, actual[0], 12); Assert.Equal(Math.Sqrt(128) * .8, actual[1], 12);
        Assert.All(actual.Skip(2), value => Assert.Equal(0, value));
    }

    [Fact]
    public void NoTrainingDataReturnsExplicitlyUnassignedOwnedResult()
    {
        using var model = new Model(); var api = new Community1Clusterer(model.Path);
        var result = api.Cluster(new[] { Vector(0), Missing() }, Mask(1, 10, 2, (_, _, _) => false), CancellationToken.None);
        Assert.Equal(new[] { -2, -2 }, result.Labels); Assert.Empty(result.Centroids); Assert.Equal(0, result.TrainingEmbeddings);
        Assert.Throws<NotSupportedException>(() => ((IList<int>)result.Labels)[0] = 0);
        var activeMissing = api.Cluster(new[] { Missing() }, Mask(1, 10, 1, (_, _, _) => true), CancellationToken.None);
        Assert.Equal(new[] { -2 }, activeMissing.Labels); Assert.Empty(activeMissing.Centroids);
    }

    [Theory]
    [InlineData(1, 0)] [InlineData(2, 1)] [InlineData(3, 1)]
    public void CleanFrameTrainingThresholdIncludesExactlyTwentyPercent(int cleanFrames, int expected)
    {
        using var model = new Model(); var api = new Community1Clusterer(model.Path);
        var result = api.Cluster(new[] { Vector(0), Missing() }, Mask(1, 10, 2, (_, t, s) => s == 0 || t >= cleanFrames), CancellationToken.None);
        Assert.Equal(expected, result.TrainingEmbeddings);
        Assert.Equal(expected == 0 ? -2 : 0, result.Labels[0]); Assert.Equal(-2, result.Labels[1]);
    }

    [Fact]
    public void SingleTrainingVectorAssignsOnlyActiveValidRowsAndOwnsCentroid()
    {
        using var model = new Model(); var api = new Community1Clusterer(model.Path);
        var values = new[] { Vector(0), Vector(1), Missing() };
        var result = api.Cluster(values, Mask(1, 10, 3, (_, _, s) => s == 0), CancellationToken.None);
        Assert.Equal(new[] { 0, -2, -2 }, result.Labels); Assert.Single(result.Centroids); Assert.Equal(1, result.Centroids[0][0]);
        Assert.Throws<NotSupportedException>(() => ((IList<double>)result.Centroids[0])[0] = 10);
        api.Cluster(new[] { Vector(2) }, Mask(1, 10, 1, (_, _, _) => true), CancellationToken.None);
        Assert.Equal(1, result.Centroids[0][0]); Assert.Equal(0, result.Centroids[0][2]);
    }

    [Fact]
    public void CompleteClusteringIsDeterministicAndIndependentAcrossCalls()
    {
        using var model = new Model(); var api = new Community1Clusterer(model.Path);
        var embeddings = new[] { Vector(0), Vector(1), Vector(1), Vector(0) };
        var activity = Mask(2, 10, 2, (_, t, s) => s == t / 5); var before = activity.ToArray();
        var result = api.Cluster(embeddings, activity, CancellationToken.None);
        Assert.Equal(4, result.TrainingEmbeddings); Assert.NotEmpty(result.Centroids);
        Assert.Throws<OperationCanceledException>(() => api.Cluster(embeddings, activity, new CancellationToken(true)));
        Parallel.For(0, 4, _ => { var next = api.Cluster(embeddings, activity, CancellationToken.None); Assert.Equal(result.Labels, next.Labels); Assert.Equal(result.Centroids.SelectMany(x => x), next.Centroids.SelectMany(x => x)); });
        Assert.Equal(before, activity.ToArray()); Assert.Equal(1, embeddings[0].Values[0]);
    }

    [Fact]
    public void InvalidInputsFailWithoutContaminatingLaterRequests()
    {
        using var model = new Model(); var api = new Community1Clusterer(model.Path); var mask = Mask(1, 10, 1, (_, _, _) => true);
        Assert.Throws<ArgumentException>(() => api.Cluster(Array.Empty<WeSpeakerEmbedding>(), mask, CancellationToken.None));
        Assert.Throws<ArgumentException>(() => api.Cluster(new[] { Vector(0) }, new DenseTensor<bool>(new[] { 10 }), CancellationToken.None));
        Assert.Throws<ArgumentException>(() => api.Cluster(new[] { Vector(0) }, Mask(1, 10, 4, (_, _, _) => false), CancellationToken.None));
        Assert.Throws<ArgumentException>(() => api.Cluster(new[] { new WeSpeakerEmbedding(new float[256], WeSpeakerEmbeddingStatus.Completed, 13, 13) }, mask, CancellationToken.None));
        var bad = new float[256]; bad[0] = float.NaN;
        Assert.Throws<ArgumentException>(() => api.Cluster(new[] { new WeSpeakerEmbedding(bad, WeSpeakerEmbeddingStatus.Completed, 13, 13) }, mask, CancellationToken.None));
        Assert.Throws<ArgumentException>(() => api.Cluster(new[] { new WeSpeakerEmbedding(new float[1], WeSpeakerEmbeddingStatus.InsufficientFrames, 1, 1) }, mask, CancellationToken.None));
        Assert.Single(api.Cluster(new[] { Vector(0) }, mask, CancellationToken.None).Centroids);
    }

    [Fact]
    public void InvalidPreparedAssetsAreRejected()
    {
        using var model = new Model(); string original = File.ReadAllText(model.Path);
        File.WriteAllText(model.Path, original.Replace("\"schema\":1", "\"schema\":2", StringComparison.Ordinal));
        Assert.Throws<InvalidDataException>(() => new Community1Clusterer(model.Path));
        File.WriteAllText(model.Path, "{}"); Assert.Throws<KeyNotFoundException>(() => new Community1Clusterer(model.Path));
    }

    [Fact]
    public void PublicRequestAndTrainingLimitsRejectBeforeHierarchyAllocation()
    {
        using var model = new Model(); var api = new Community1Clusterer(model.Path); var vector = Vector(0);
        var tooMany = Enumerable.Repeat(vector, Community1Clusterer.MaximumEmbeddings + 1).ToArray();
        Assert.Throws<ArgumentException>(() => api.Cluster(tooMany, Mask(tooMany.Length, 1, 1, (_, _, _) => true), CancellationToken.None));
        var tooManyTraining = Enumerable.Repeat(vector, Community1Clusterer.MaximumTrainingEmbeddings + 1).ToArray();
        var failure = Assert.Throws<ArgumentException>(() => api.Cluster(tooManyTraining, Mask(tooManyTraining.Length, 1, 1, (_, _, _) => true), CancellationToken.None));
        Assert.Contains("4096 training", failure.Message);
        Assert.Throws<ArgumentException>(() => Hierarchy(new float[4097], 4097, 1, .6));
    }
}
