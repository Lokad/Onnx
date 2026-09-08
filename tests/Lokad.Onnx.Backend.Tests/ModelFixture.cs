namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Mechanics shared by the local-model test classes: repository-relative model
/// lookup, documented lane gating, execute-or-fail, and the compact
/// shape/finiteness/mean/spot oracle checks. Per-model oracle numbers stay in
/// the test classes; only the mechanics live here.
/// </summary>
static class ModelFixture
{
    public const string LaneVariable = "LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS";

    public static string? FindModelPath(params string[] parts)
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir is not null)
        {
            var candidate = Path.Combine(new[] { dir.FullName }.Concat(parts).ToArray());
            if (File.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        return null;
    }

    public static string RequireModelOrSkip(string displayName, params string[] parts)
    {
        var modelPath = FindModelPath(parts);
        if (modelPath is null && Environment.GetEnvironmentVariable(LaneVariable) == "1")
        {
            Assert.Fail(displayName + " model requested via " + LaneVariable + "=1 but not found at " + string.Join("/", parts) + ".");
        }
        Skip.If(modelPath is null, displayName + " model not present; set " + LaneVariable + "=1 to require it.");
        return modelPath!;
    }

    public static ComputationalGraph LoadRequiredModel(string displayName, params string[] parts)
    {
        var graph = OnnxImport.Load(RequireModelOrSkip(displayName, parts));
        Assert.NotNull(graph);
        return graph!;
    }

    public static void AssertExecuted(ComputationalGraph graph, bool ok)
    {
        if (!ok)
        {
            var details = $"Failure at node '{graph.LastFailedNodeName}' ({graph.LastFailedNodeOp}): {graph.LastErrorMessage}";
            Assert.True(ok, details);
        }
    }

    public static float[] CheckedOutput(ComputationalGraph graph, string outputName)
    {
        var output = (Tensor<float>)graph.Outputs[outputName];
        return CheckedValues(output.ToArray(), outputName);
    }

    public static float[] CheckedFirstOutput(ComputationalGraph graph)
    {
        var output = (Tensor<float>)graph.Outputs.Values.First();
        return CheckedValues(output.ToArray(), "first-output");
    }

    static float[] CheckedValues(float[] values, string what)
    {
        foreach (var v in values)
        {
            Assert.False(float.IsNaN(v) || float.IsInfinity(v), "Non-finite value in " + what + ".");
        }
        return values;
    }

    public static void AssertMean(float[] values, double expected, double tolerance, string what)
    {
        double sum = 0;
        foreach (var v in values) sum += v;
        Assert.True(Math.Abs(sum / values.Length - expected) < tolerance, "Mean drift vs ORT reference in " + what + ".");
    }

    public static void AssertSpots(float[] values, int[] spots, float[] expected, double tolerance, string what)
    {
        Assert.Equal(spots.Length, expected.Length);
        for (int i = 0; i < spots.Length; i++)
            Assert.True(Math.Abs(values[spots[i]] - expected[i]) < tolerance, "Spot " + spots[i] + " drift vs ORT reference in " + what + ".");
    }
}
