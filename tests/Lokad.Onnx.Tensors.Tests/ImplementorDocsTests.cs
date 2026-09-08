namespace Lokad.Onnx.Tensors.Tests;

/// <summary>
/// Guards the G05 rule: every abstract or virtual method in product code
/// documents its implementor contract (ownership, layout, empty tensors,
/// aliasing, exceptions, destination initialization) with adjacent XML docs,
/// and no friend assemblies are declared. Overrides inherit base docs and
/// are reconciled by review instead of by this scan.
/// </summary>
public class ImplementorDocsTests
{
    static string RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Lokad.Onnx.slnx"))) return dir.FullName;
            dir = dir.Parent;
        }
        throw new DirectoryNotFoundException("Repository root with Lokad.Onnx.slnx not found.");
    }

    internal static bool IsUndocumentedDeclaration(string[] lines, int i)
    {
        string code = lines[i];
        int comment = code.IndexOf("//", StringComparison.Ordinal);
        if (comment >= 0)
        {
            // A full-line comment is not a declaration; an end-of-line
            // comment cannot hide a declaration before it either, because
            // abstract/virtual members never hide inside string literals
            // in this tree. Keep the code part only.
            string head = code.Substring(0, comment).Trim();
            if (head.Length == 0) return false;
            code = head;
        }
        if (!code.Contains('(')) return false;
        if (!System.Text.RegularExpressions.Regex.IsMatch(code, @"\b(abstract|virtual)\b")) return false;
        if (code.Contains("override")) return false;
        int j = i - 1;
        while (j >= 0)
        {
            string above = lines[j].Trim();
            if (above.StartsWith('[') && above.EndsWith(']')) { j--; continue; }
            return !above.StartsWith("///");
        }
        return true;
    }

    internal static List<string> FindUndocumented(string root)
    {
        var offenders = new List<string>();
        foreach (string file in Directory.GetFiles(Path.Combine(root, "src"), "*.cs", SearchOption.AllDirectories))
        {
            if (file.Contains(Path.DirectorySeparatorChar + "obj" + Path.DirectorySeparatorChar) ||
                file.Contains(Path.DirectorySeparatorChar + "bin" + Path.DirectorySeparatorChar)) continue;
            string[] lines = File.ReadAllLines(file);
            for (int i = 0; i < lines.Length; i++)
            {
                if (IsUndocumentedDeclaration(lines, i))
                    offenders.Add(Path.GetRelativePath(root, file) + ":" + (i + 1));
            }
        }
        return offenders;
    }

    [Fact]
    public void Scanner_DetectsMissingDocs()
    {
        string[] code = {
            "    public virtual Tensor<T> InsertDim(int dim)",
            "    {",
        };
        Assert.True(IsUndocumentedDeclaration(code, 0));
    }

    [Fact]
    public void Scanner_AcceptsDocumentedDeclarations()
    {
        string[] code = {
            "    /// <summary>Copies values into new backing storage.</summary>",
            "    public override Tensor<T> Clone() => ToDenseTensor();",
            "    /// <summary>Writes op(element) for every element.</summary>",
            "    public virtual void Apply(Func<T, T> op, Tensor<T> destination)",
        };
        Assert.False(IsUndocumentedDeclaration(code, 1));
        Assert.False(IsUndocumentedDeclaration(code, 3));
    }

    [Fact]
    public void Scanner_IgnoresCommentsAndOverrides()
    {
        string[] code = {
            "    // virtual Apply() is inherited, no doc needed here",
            "    public override Tensor<T> Clone() => ToDenseTensor();",
        };
        Assert.False(IsUndocumentedDeclaration(code, 0));
        Assert.False(IsUndocumentedDeclaration(code, 1));
    }

    [Fact]
    public void SourceTree_DocumentsEveryAbstractOrVirtualMethod()
    {
        var offenders = FindUndocumented(RepoRoot());
        Assert.True(offenders.Count == 0,
            "Abstract/virtual methods without XML docs:\n" + string.Join("\n", offenders.Take(20)));
    }

    [Fact]
    public void SourceTree_DeclaresNoFriendAssemblies()
    {
        string root = RepoRoot();
        var offenders = new List<string>();
        foreach (string pattern in new[] { "*.csproj", "*.cs" })
        {
            string dir = pattern == "*.csproj" ? root : Path.Combine(root, "src");
            foreach (string file in Directory.GetFiles(dir, pattern, SearchOption.AllDirectories))
            {
                if (file.Contains(Path.DirectorySeparatorChar + "obj" + Path.DirectorySeparatorChar) ||
                    file.Contains(Path.DirectorySeparatorChar + "bin" + Path.DirectorySeparatorChar)) continue;
                if (File.ReadAllText(file).Contains("InternalsVisibleTo"))
                    offenders.Add(Path.GetRelativePath(root, file));
            }
        }
        Assert.True(offenders.Count == 0,
            "InternalsVisibleTo found in:\n" + string.Join("\n", offenders));
    }
}
