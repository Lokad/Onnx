namespace Lokad.Onnx.Tensors.Tests;

using Lokad.Onnx.Tests.Support;

/// <summary>
/// Guards the G05 rule: every abstract or virtual method in product code
/// documents its implementor contract (ownership, layout, empty tensors,
/// aliasing, exceptions, destination initialization) with adjacent XML docs,
/// and exactly one approved friend edge exists (Backend.Tests in Global.cs).
/// Overrides inherit base docs and are reconciled by review instead of by
/// this scan.
/// </summary>
public class ImplementorDocsTests
{

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
        foreach (string file in TestSupport.ProductSources())
        {
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
        var offenders = FindUndocumented(TestSupport.RepoRoot());
        Assert.True(offenders.Count == 0,
            "Abstract/virtual methods without XML docs:\n" + string.Join("\n", offenders.Take(20)));
    }

    internal static readonly string ApprovedFriendFile = "src/Lokad.Onnx/Global.cs";
    internal const string ApprovedFriendAssembly = "Lokad.Onnx.Backend.Tests";

    internal static bool IsApprovedFriendFile(string relativePath, string content)
    {
        if (!relativePath.Equals(ApprovedFriendFile, StringComparison.Ordinal)) return false;
        var matches = System.Text.RegularExpressions.Regex.Matches(content, "InternalsVisibleTo\\(\\s*\"([^\"]+)\"\\s*\\)");
        return matches.Count == 1 && matches[0].Groups[1].Value == ApprovedFriendAssembly;
    }

    internal static List<string> FindFriendAssemblyViolations(string root)
    {
        var offenders = new List<string>();
        foreach (string file in MaintainedProjects(root))
        {
            string rel = Path.GetRelativePath(root, file).Replace(Path.DirectorySeparatorChar, '/');
            string text = File.ReadAllText(file);
            if (IsApprovedFriendFile(rel, text)) continue;
            if (text.Contains("InternalsVisibleTo")) offenders.Add(rel);
        }
        foreach (string file in TestSupport.ProductSources())
        {
            string rel = Path.GetRelativePath(root, file).Replace(Path.DirectorySeparatorChar, '/');
            string text = File.ReadAllText(file);
            if (IsApprovedFriendFile(rel, text)) continue;
            if (text.Contains("InternalsVisibleTo")) offenders.Add(rel);
        }
        return offenders;
    }

    static IEnumerable<string> MaintainedProjects(string root)
    {
        // Maintained build surface only: ignored research checkouts under
        // artifacts/, external/ or eng scratch must not control the result.
        foreach (string dir in new[] { "src", "tests" })
        {
            string full = Path.Combine(root, dir);
            if (!Directory.Exists(full)) continue;
            foreach (string file in Directory.GetFiles(full, "*.csproj", SearchOption.AllDirectories))
            {
                if (file.Contains(Path.DirectorySeparatorChar + "obj" + Path.DirectorySeparatorChar) ||
                    file.Contains(Path.DirectorySeparatorChar + "bin" + Path.DirectorySeparatorChar)) continue;
                yield return file;
            }
        }
        foreach (string file in Directory.GetFiles(root, "*.csproj", SearchOption.TopDirectoryOnly))
            yield return file;
    }

    [Fact]
    public void Allowlist_AcceptsSingleApprovedEdge()
    {
        string good = "[assembly: System.Runtime.CompilerServices.InternalsVisibleTo(\"Lokad.Onnx.Backend.Tests\")]";
        Assert.True(IsApprovedFriendFile(ApprovedFriendFile, good));
    }

    [Fact]
    public void Allowlist_RejectsSecondEdgeWrongAssemblyAndOtherFiles()
    {
        string two = "[assembly: InternalsVisibleTo(\"Lokad.Onnx.Backend.Tests\")]\n[assembly: InternalsVisibleTo(\"Other.Tests\")]";
        Assert.False(IsApprovedFriendFile(ApprovedFriendFile, two));
        string wrong = "[assembly: InternalsVisibleTo(\"Other.Tests\")]";
        Assert.False(IsApprovedFriendFile(ApprovedFriendFile, wrong));
        string good = "[assembly: InternalsVisibleTo(\"Lokad.Onnx.Backend.Tests\")]";
        Assert.False(IsApprovedFriendFile("src/Lokad.Onnx/Other.cs", good));
        Assert.False(IsApprovedFriendFile(ApprovedFriendFile, "// no edges here"));
    }

    [Fact]
    public void SourceTree_DeclaresNoFriendAssemblies()
    {
        var offenders = FindFriendAssemblyViolations(TestSupport.RepoRoot());
        Assert.True(offenders.Count == 0,
            "Unapproved InternalsVisibleTo found in:\n" + string.Join("\n", offenders));
    }
}
