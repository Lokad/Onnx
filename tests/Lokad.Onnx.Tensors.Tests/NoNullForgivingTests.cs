
namespace Lokad.Onnx.Tensors.Tests;

using Lokad.Onnx.Tests.Support;

/// <summary>
/// Guards the G02 rule: product code contains no null-forgiving operators.
/// Absence is expressed with nullable annotations and narrowing instead; the
/// compiler verifies it with zero nullable warnings. The scan counts postfix
/// suppressions only, ignoring logical negation, inequalities and string or
/// comment contents.
/// </summary>
public class NoNullForgivingTests
{

    internal static List<string> FindSuppressions(string code)
    {
        var hits = new List<string>();
        string clean = StripNoise(code);
        for (int i = 0; i < clean.Length; i++)
        {
            if (clean[i] != '!') continue;
            char prev = i > 0 ? clean[i - 1] : '\0';
            char next = i + 1 < clean.Length ? clean[i + 1] : '\0';
            if (prev == '=' || prev == '!' || next == '=') continue;
            int p = i - 1;
            while (p >= 0 && (clean[p] == ' ' || clean[p] == '\t' || clean[p] == '\r')) p--;
            if (p < 0) continue;
            char pc = clean[p];
            if (!(char.IsLetterOrDigit(pc) || pc == '_' || pc == ')' || pc == ']' || pc == '}' || pc == '?' || pc == '"' || pc == '\'')) continue;
            int q = i + 1;
            while (q < clean.Length && (clean[q] == ' ' || clean[q] == '\t' || clean[q] == '\r')) q++;
            char nc = q < clean.Length ? clean[q] : '\n';
            if (nc == '(') continue;
            int s = Math.Max(0, i - 40);
            hits.Add(clean.Substring(s, i - s).Trim() + "!");
        }
        return hits;
    }

    static string StripNoise(string code)
    {
        var sb = new System.Text.StringBuilder(code.Length);
        int i = 0, n = code.Length;
        while (i < n)
        {
            char c = code[i];
            if (c == '\n') { sb.Append(c); i++; continue; }
            if (c == '/' && i + 1 < n && code[i + 1] == '/')
            {
                while (i < n && code[i] != '\n') i++;
                continue;
            }
            if (c == '/' && i + 1 < n && code[i + 1] == '*')
            {
                i += 2;
                while (i + 1 < n && !(code[i] == '*' && code[i + 1] == '/')) i++;
                i += 2;
                continue;
            }
            if (c == '@' && i + 1 < n && code[i + 1] == '"')
            {
                i += 2;
                while (i < n)
                {
                    if (code[i] == '"' && i + 1 < n && code[i + 1] == '"') { i += 2; continue; }
                    if (code[i] == '"') { i++; break; }
                    i++;
                }
                sb.Append("\"\"");
                continue;
            }
            if (c == '"')
            {
                i++;
                while (i < n && code[i] != '"') { if (code[i] == '\\') i++; i++; }
                i++;
                sb.Append("\"\"");
                continue;
            }
            if (c == '\'')
            {
                i++;
                while (i < n && code[i] != '\'') { if (code[i] == '\\') i++; i++; }
                i++;
                sb.Append("''");
                continue;
            }
            sb.Append(c);
            i++;
        }
        return sb.ToString();
    }

    [Fact]
    public void Scanner_DetectsSuppressions()
    {
        Assert.Single(FindSuppressions("var x = graph!.Reset();"));
        Assert.Single(FindSuppressions("return t!.Text;"));
        Assert.Equal(2, FindSuppressions("use(dx!, dy!)").Count);
    }

    [Fact]
    public void Scanner_IgnoresNegationAndStrings()
    {
        Assert.Empty(FindSuppressions("return !(a == b);"));
        Assert.Empty(FindSuppressions("if (x != null && y == 2) {}"));
        Assert.Empty(FindSuppressions(@"var r = new Regex(""(?!x)"");"));
        Assert.Empty(FindSuppressions("bool b = !ready;"));
    }

    [Fact]
    public void SourceTree_HasNoNullForgivingOperators()
    {
        string root = TestSupport.RepoRoot();
        var offenders = new List<string>();
        foreach (string file in TestSupport.ProductSources())
        {
            var hits = FindSuppressions(File.ReadAllText(file));
            foreach (string h in hits) offenders.Add(Path.GetRelativePath(root, file) + ": " + h);
        }
        Assert.True(offenders.Count == 0, "Null-forgiving operators found:\n" + string.Join("\n", offenders.Take(20)));
    }
}
