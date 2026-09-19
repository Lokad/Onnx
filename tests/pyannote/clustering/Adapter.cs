// Qualification only: call the actual compiled internal primitives without widening the product API.
using System.Reflection;
using System.Runtime.ExceptionServices;
using Lokad.Onnx;

static class ReflectionAdapter
{
    internal static Type Type(string name) => typeof(Community1Clusterer).Assembly.GetType("Lokad.Onnx." + name, true) ?? throw new InvalidOperationException(name);
    internal static object Invoke(string type, string method, object? target, params object[] args)
    {
        try { return Type(type).GetMethod(method, BindingFlags.Static | BindingFlags.Instance | BindingFlags.NonPublic)?.Invoke(target, args) ?? throw new InvalidOperationException(method); }
        catch (TargetInvocationException e) when (e.InnerException is not null) { ExceptionDispatchInfo.Capture(e.InnerException).Throw(); throw; }
    }
    internal static T Property<T>(object target, string name) => (T)(target.GetType().GetProperty(name)?.GetValue(target) ?? throw new InvalidOperationException(name));
}
sealed class Community1Parameters
{
    readonly object parameters;
    internal Community1Parameters(string path) => parameters = Activator.CreateInstance(ReflectionAdapter.Type("Community1Parameters"), BindingFlags.Instance | BindingFlags.NonPublic, null, new object[] { path }, null) ?? throw new InvalidOperationException();
    internal double[] Phi => (double[])(parameters.GetType().GetField("Phi", BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(parameters) ?? throw new InvalidOperationException());
    internal double[] Transform(float[] vectors, int count, CancellationToken cancellation) => (double[])ReflectionAdapter.Invoke("Community1Parameters", "Transform", parameters, vectors, count, cancellation);
}
sealed record HierarchyResult(int[] Labels, double[] Distances);
sealed record RefinementResult(double[] Responsibilities, double[] Priors, double[] Objective, double[] Alpha, double[] InversePrecision);
static class Community1Math
{
    internal static float[] Normalize(float[] vectors, int count, int dimensions) => (float[])ReflectionAdapter.Invoke("Community1Math", "Normalize", null, vectors, count, dimensions);
    internal static HierarchyResult Hierarchy(float[] points, int count, int dimensions, double threshold, CancellationToken cancellation)
    {
        var r = ReflectionAdapter.Invoke("Community1Math", "Hierarchy", null, points, count, dimensions, threshold, cancellation);
        return new HierarchyResult(ReflectionAdapter.Property<int[]>(r, "Labels"), ReflectionAdapter.Property<double[]>(r, "Distances"));
    }
    internal static RefinementResult Refine(double[] x, int count, int dimensions, double[] phi, int[] labels, CancellationToken cancellation)
    {
        var r = ReflectionAdapter.Invoke("Community1Math", "Refine", null, x, count, dimensions, phi, labels, cancellation);
        return new RefinementResult(ReflectionAdapter.Property<double[]>(r, "Responsibilities"), ReflectionAdapter.Property<double[]>(r, "Priors"), ReflectionAdapter.Property<double[]>(r, "Objective"), ReflectionAdapter.Property<double[]>(r, "Alpha"), ReflectionAdapter.Property<double[]>(r, "InversePrecision"));
    }
    internal static int[] Match(double[] scores, int rows, int columns) => (int[])ReflectionAdapter.Invoke("Community1Math", "Match", null, scores, rows, columns);
}
