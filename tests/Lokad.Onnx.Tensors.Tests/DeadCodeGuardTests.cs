namespace Lokad.Onnx.Tensors.Tests;

using Lokad.Onnx.Tests.Support;

/// <summary>
/// Guards the G06 rule: dead implementation files, commented-out code and
/// stale prehistory identifiers stay out of product sources. The two
/// remaining TODOs are triaged future work (API design or framework support),
/// so the test pins their exact list instead of demanding zero.
/// </summary>
public class DeadCodeGuardTests
{
    internal static List<string> FindBannedIdentifiers(string root)
    {
        string[] banned = { "NDArray", "ViewInfo", "OriginalShape", "IsBroadcasted", "GetConv2DDefaultPad", "class BLAS" };
        var offenders = new List<string>();
        foreach (string file in TestSupport.ProductSources())
        {
            string[] lines = File.ReadAllLines(file);
            for (int i = 0; i < lines.Length; i++)
            {
                foreach (string b in banned)
                {
                    if (lines[i].Contains(b, StringComparison.Ordinal))
                        offenders.Add(Path.GetRelativePath(root, file) + ":" + (i + 1) + ": " + b);
                }
            }
        }
        return offenders;
    }

    internal static List<string> FindTodos(string root)
    {
        var todos = new List<string>();
        foreach (string file in TestSupport.ProductSources())
        {
            string[] lines = File.ReadAllLines(file);
            for (int i = 0; i < lines.Length; i++)
            {
                int t = lines[i].IndexOf("TODO", StringComparison.Ordinal);
                if (t >= 0)
                    todos.Add(Path.GetRelativePath(root, file) + ":" + (i + 1) + ": " + lines[i].Trim());
            }
        }
        return todos;
    }

    [Fact]
    public void Scanner_FindsBannedIdentifiers()
    {
        string root = TestSupport.RepoRoot();
        var offenders = FindBannedIdentifiers(root);
        Assert.True(offenders.Count == 0,
            "Dead identifiers found:\n" + string.Join("\n", offenders.Take(20)));
    }

    [Fact]
    public void DeadFiles_AreAbsent()
    {
        string root = TestSupport.RepoRoot();
        var present = new List<string>();
        foreach (string rel in new[] {
            Path.Combine("src", "Lokad.Onnx", "BLAS.cs"),
            Path.Combine("src", "Lokad.Onnx.Tensors", "Global.cs") })
        {
            if (File.Exists(Path.Combine(root, rel))) present.Add(rel);
        }
        Assert.True(present.Count == 0, "Dead files present:\n" + string.Join("\n", present));
    }

    [Fact]
    public void Todos_MatchTriagedList()
    {
        var todos = FindTodos(TestSupport.RepoRoot());
        var expected = new HashSet<string>(StringComparer.Ordinal);
        foreach (string t in todos)
        {
            string file = t.Split(':')[0];
            if (file.EndsWith("DenseTensor.cs") && t.Contains("Span.IndexOf")) expected.Add(t);
            else if (file.EndsWith("HardwareIntrinsics.cs") && t.Contains("MOVBE")) expected.Add(t);
        }
        var unexpected = todos.Where(t => !expected.Contains(t)).ToList();
        Assert.True(unexpected.Count == 0,
            "Untriaged TODOs found:\n" + string.Join("\n", unexpected.Take(20)));
        Assert.Equal(2, todos.Count);
    }
}
