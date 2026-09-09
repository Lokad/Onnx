namespace Lokad.Onnx.Tensors.Tests;

using Lokad.Onnx.Tests.Support;

/// <summary>
/// Guards the G06 rule: new TODOs in product sources need explicit triage.
/// The filter below names the accepted ones with their reason; anything else
/// fails naming the offender. Resolving a triaged TODO never breaks this test.
/// </summary>
public class DeadCodeGuardTests
{
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
    }
}
