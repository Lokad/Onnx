
namespace Lokad.Onnx.Tensors.Tests;

using Lokad.Onnx.Tests.Support;

/// <summary>
/// Guards the G01 rule: no first-party method, constructor, record positional
/// or local function may declare a default parameter value. Callers state
/// every argument; the compiler enforces it. The scan flags any parameter
/// chunk whose left side carries a type plus a name before a top-level equals
/// sign, which catches declarations with or without accessibility keywords
/// while ignoring named arguments, assignments, lambdas, loops and usings.
/// </summary>
public class NoOptionalParametersTests
{

    internal static List<string> FindOptionalParameters(string code)
    {
        var hits = new List<string>();
        string clean = StripNoise(code);
        foreach (var (name, args) in EachParenthesized(clean))
        {
            if (IsKeyword(name)) continue;
            foreach (var chunk in SplitTopLevel(args))
            {
                int eq = TopLevelEquals(chunk);
                if (eq < 0) continue;
                string left = chunk.Substring(0, eq);
                if (IsTypedName(left)) hits.Add(name + "(" + chunk.Trim() + ")");
            }
        }
        return hits;
    }

    static readonly HashSet<string> Keywords = new HashSet<string>(StringComparer.Ordinal)
    {
        "for", "foreach", "while", "if", "else", "switch", "using", "lock",
        "catch", "fixed", "checked", "unchecked", "return", "new", "typeof",
        "nameof", "sizeof", "default", "throw", "await", "where", "get", "set",
        "add", "remove", "operator", "true", "false", "null", "is", "as",
        "base", "this",
    };

    static bool IsKeyword(string name) => Keywords.Contains(name);

    static bool IsTypedName(string left)
    {
        string t = left.Trim();
        if (t.Length == 0 || t.Contains("(") || t.Contains(";") || t.Contains("{") || t.Contains("}") || t.Contains("=>")) return false;
        var words = System.Text.RegularExpressions.Regex.Matches(t, @"[A-Za-z_]\w*");
        return words.Count >= 2;
    }

    static int TopLevelEquals(string chunk)
    {
        int da = 0, dp = 0, db = 0, dc = 0;
        for (int i = 0; i < chunk.Length; i++)
        {
            char ch = chunk[i];
            if (ch == '<') da++;
            else if (ch == '>') da = Math.Max(0, da - 1);
            else if (ch == '(') dp++;
            else if (ch == ')') dp = Math.Max(0, dp - 1);
            else if (ch == '[') db++;
            else if (ch == ']') db = Math.Max(0, db - 1);
            else if (ch == '{') dc++;
            else if (ch == '}') dc = Math.Max(0, dc - 1);
            else if (ch == '=' && da == 0 && dp == 0 && db == 0 && dc == 0)
            {
                char prev = i > 0 ? chunk[i - 1] : '\0';
                char next = i + 1 < chunk.Length ? chunk[i + 1] : '\0';
                if (prev == '=' || prev == '!' || prev == '<' || prev == '>' || next == '=' || next == '>') continue;
                if (prev == '?' && (i < 2 || chunk[i - 2] != '?')) continue;
                return i;
            }
        }
        return -1;
    }

    static IEnumerable<(string name, string args)> EachParenthesized(string code)
    {
        var m = System.Text.RegularExpressions.Regex.Matches(code, @"(?<![\w.])([A-Za-z_]\w*)(?:<[^()]*>)?\s*\(");
        foreach (System.Text.RegularExpressions.Match mm in m)
        {
            int oi = mm.Index + mm.Length - 1;
            int depth = 0, j = oi;
            while (j < code.Length)
            {
                if (code[j] == '(') depth++;
                else if (code[j] == ')') { depth--; if (depth == 0) break; }
                j++;
            }
            if (j < code.Length) yield return (mm.Groups[1].Value, code.Substring(oi + 1, j - oi - 1));
        }
    }

    static IEnumerable<string> SplitTopLevel(string s)
    {
        int da = 0, dp = 0, db = 0, dc = 0, start = 0;
        for (int i = 0; i < s.Length; i++)
        {
            char ch = s[i];
            if (ch == '<') da++;
            else if (ch == '>') da = Math.Max(0, da - 1);
            else if (ch == '(') dp++;
            else if (ch == ')') dp = Math.Max(0, dp - 1);
            else if (ch == '[') db++;
            else if (ch == ']') db = Math.Max(0, db - 1);
            else if (ch == '{') dc++;
            else if (ch == '}') dc = Math.Max(0, dc - 1);
            else if (ch == ',' && da == 0 && dp == 0 && db == 0 && dc == 0)
            {
                yield return s.Substring(start, i - start);
                start = i + 1;
            }
        }
        yield return s.Substring(start);
    }

    static string StripNoise(string code)
    {
        var sb = new System.Text.StringBuilder(code.Length);
        int i = 0, n = code.Length;
        bool atLineStart = true;
        while (i < n)
        {
            char c = code[i];
            if (c == '\n') { sb.Append(c); i++; atLineStart = true; continue; }
            if (c == ' ' || c == '\t' || c == '\r') { sb.Append(c); i++; continue; }
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
                atLineStart = false;
                continue;
            }
            if (c == '"')
            {
                i++;
                while (i < n && code[i] != '"') { if (code[i] == '\\') i++; i++; }
                i++;
                sb.Append("\"\"");
                atLineStart = false;
                continue;
            }
            if (c == '\'')
            {
                i++;
                while (i < n && code[i] != '\'') { if (code[i] == '\\') i++; i++; }
                i++;
                sb.Append("''");
                atLineStart = false;
                continue;
            }
            if (atLineStart && c == '[')
            {
                int depth = 0;
                while (i < n)
                {
                    if (code[i] == '[') depth++;
                    else if (code[i] == ']') { depth--; if (depth == 0) { i++; break; } }
                    i++;
                }
                continue;
            }
            atLineStart = false;
            sb.Append(c);
            i++;
        }
        return sb.ToString();
    }

    [Fact]
    public void Scanner_DetectsDeclarationDefaults()
    {
        Assert.Single(FindOptionalParameters("void M(int x = 0) {}"));
        Assert.Single(FindOptionalParameters("public Foo(string? s = null) {}"));
        Assert.Single(FindOptionalParameters("void Indent(StringBuilder b, int t, int s = 4) {}"));
    }

    [Fact]
    public void Scanner_IgnoresCallsAndStatements()
    {
        Assert.Empty(FindOptionalParameters("Foo(bar: 1);"));
        Assert.Empty(FindOptionalParameters("for (int i = 0; i < n; i++) {}"));
        Assert.Empty(FindOptionalParameters("using (var s = Open()) {}"));
        Assert.Empty(FindOptionalParameters("if ((line = Next()) != null) {}"));
        Assert.Empty(FindOptionalParameters("M(x = 1);"));
    }

    [Fact]
    public void SourceTree_HasNoOptionalParameters()
    {
        string root = TestSupport.RepoRoot();
        var offenders = new List<string>();
        foreach (string file in TestSupport.SourceFiles("src", "tests"))
        {
            var hits = FindOptionalParameters(File.ReadAllText(file));
            foreach (string h in hits) offenders.Add(Path.GetRelativePath(root, file) + ": " + h);
        }
        Assert.True(offenders.Count == 0, "Optional parameters found:\n" + string.Join("\n", offenders.Take(20)));
    }
}
