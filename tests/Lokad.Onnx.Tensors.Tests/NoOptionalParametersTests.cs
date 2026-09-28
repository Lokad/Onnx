
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
    // Immutable inputs to closed experiments, not the current test-project
    // implementations. The latter use explicit overloads and are scanned below.
    // Keep the historical bytes reproducible without exempting a directory or
    // accepting any new declaration in these files.
    static readonly IReadOnlyDictionary<string, string> ArchivedSources = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["tests/pyannote/request-contexts/PipelineRequestTests.cs"] = "8409181643c4e7f405ac4a8fb39a9255933aa9924d9c2aecae20926f2ac49583",
        ["tests/whisper/weight-sharing/WhisperDecoderWeightsTests.cs"] = "4d8ca0bfb98511eec9709225479c35ba6ab8c81423325e9a12b82ca51baba9e0",
        // Exact retained inputs reviewed in tests/Shared/closed-experiment-sources-20260928.json.
        ["tests/parakeet/dense-scalar-where-balanced-codegen/QualifiedSetup.cs"] = "613636fa19b0a1409fd467e79737a1fb0fd56c7b6fb6c5b5a068b86f842047a1",
        ["tests/parakeet/dense-scalar-where-numerics-amd/Contracts.cs"] = "9f03a65e6bc82e825d0168c8b0e2ea92afdda2f3498da4f380883e56dd515cd9",
        ["tests/parakeet/dense-scalar-where-numerics-amd/Driver.cs"] = "a2596635854142679b126845bf8cb779d5650aa66fdee8ad330f5c76844b5bfa",
        ["tests/parakeet/dense-scalar-where-stability-amd/QualifiedSetup.cs"] = "613636fa19b0a1409fd467e79737a1fb0fd56c7b6fb6c5b5a068b86f842047a1",
        ["tests/parakeet/first-use-kernels-numerics/Driver.cs"] = "4e48d142c0ca2126ce7f5ac22796f1df5e26bc531d297ff8ed06c4a65802fcbb",
        ["tests/parakeet/isolated-short-kernels-numerics/Driver.cs"] = "24960c1e5a293e9922f9a10e95be1fd6c9809fe8a62b159ccf3a9912d8be7031",
        ["tests/parakeet/last-axis-pad-source/LastAxisPadTests.cs"] = "f2de068bf33a8dbb1f096f1fca3a81ad48cf6e7df8f2ab36f96cbb71457d241e",
        ["tests/parakeet/observed-dense-where-numerics-amd/Contracts.cs"] = "9f03a65e6bc82e825d0168c8b0e2ea92afdda2f3498da4f380883e56dd515cd9",
        ["tests/parakeet/observed-dense-where-numerics-amd/Driver.cs"] = "a2596635854142679b126845bf8cb779d5650aa66fdee8ad330f5c76844b5bfa",
        ["tests/parakeet/ordered-wide-blocks-numerics/Driver.cs"] = "f220918991347f4880333d4ef5202ac3943872d337e65dae73df2cef0e0134a4",
        ["tests/parakeet/pad-dispatch-source/LastAxisPadTests.cs"] = "f2de068bf33a8dbb1f096f1fca3a81ad48cf6e7df8f2ab36f96cbb71457d241e",
        ["tests/parakeet/pad-first-use-source/LastAxisPadTests.cs"] = "f2de068bf33a8dbb1f096f1fca3a81ad48cf6e7df8f2ab36f96cbb71457d241e",
        ["tests/parakeet/prepared-recurrence-calls-amd-v2/Calls.cs"] = "185fb438f37d6b007ad7c59b32c6e925f066051357a3171b18d73fe0e0cae69f",
        ["tests/parakeet/prepared-recurrence-calls-amd/Calls.cs"] = "bffbaa2802216b2d56b99761a0ce0a4cef464bf080e446dd9467bb013917effd",
        ["tests/parakeet/prepared-recurrence-source/PreparedLstmWeightsTests.cs"] = "2013db1a60aed25615087076c958a8ab7b3a19ca585607e3b22651e2b319de58",
        ["tests/parakeet/provider-where-fallback-codegen/QualifiedSetup.cs"] = "613636fa19b0a1409fd467e79737a1fb0fd56c7b6fb6c5b5a068b86f842047a1",
        ["tests/parakeet/provider-where-numerics-v2/Contracts.cs"] = "9f03a65e6bc82e825d0168c8b0e2ea92afdda2f3498da4f380883e56dd515cd9",
        ["tests/parakeet/provider-where-numerics-v2/Driver.cs"] = "e770efd3ea746508dc0fcd6281c81613db1702a52ae97343446e106598dacd11",
        ["tests/parakeet/provider-where-numerics/Contracts.cs"] = "ad76a71d3075567bf4a694bc050b41f41e8345bdd33b9bebd0715e8be1412628",
        ["tests/parakeet/provider-where-numerics/Driver.cs"] = "aa2e2616a2309560d9be87478c59b16126623fb90ea8e0dc09ff3112e03489a3",
        ["tests/parakeet/provider-where-screen/QualifiedSetup.cs"] = "613636fa19b0a1409fd467e79737a1fb0fd56c7b6fb6c5b5a068b86f842047a1",
        ["tests/parakeet/scalar-where-fallback-codegen/QualifiedSetup.cs"] = "613636fa19b0a1409fd467e79737a1fb0fd56c7b6fb6c5b5a068b86f842047a1",
        ["tests/parakeet/scalar-where-numerics-v3/Driver.cs"] = "f1e4f11161e003c081eeb49ab5d91f125819c34c79cbc65135659304fee55329",
        ["tests/parakeet/scalar-where-numerics/Driver.cs"] = "5f1761f188cc47c10b5f47c72b1dd859a3f66e6e1c0cf27df59959bf8ec3eb2c",
        ["tests/parakeet/scalar-where-screen/QualifiedSetup.cs"] = "613636fa19b0a1409fd467e79737a1fb0fd56c7b6fb6c5b5a068b86f842047a1",
        ["tests/parakeet/short-dispatch-numerics-v2/Driver.cs"] = "4e48d142c0ca2126ce7f5ac22796f1df5e26bc531d297ff8ed06c4a65802fcbb",
        ["tests/parakeet/short-dispatch-numerics/Driver.cs"] = "4e48d142c0ca2126ce7f5ac22796f1df5e26bc531d297ff8ed06c4a65802fcbb",
        ["tests/parakeet/short-wide-pack-numerics/Driver.cs"] = "4e48d142c0ca2126ce7f5ac22796f1df5e26bc531d297ff8ed06c4a65802fcbb",
        ["tests/parakeet/wide-entry-first-use-numerics/Driver.cs"] = "3e787dc9bfb89ad956c0e2602d0a005171978659634f2ab679ae5c9a7e32e736",
        ["tests/parakeet/wide-per-call-numerics/Driver.cs"] = "b7c71c643325e628b6ff77c436615437b328844bd7fb148d234f497029bda046",
        ["tests/parakeet/wide-projection-isolation-numerics-v2/Driver.cs"] = "24960c1e5a293e9922f9a10e95be1fd6c9809fe8a62b159ccf3a9912d8be7031",
        ["tests/parakeet/wide-projection-isolation-numerics/Driver.cs"] = "24960c1e5a293e9922f9a10e95be1fd6c9809fe8a62b159ccf3a9912d8be7031",
        ["tests/pyannote/winograd-contiguous-prototype/Driver.cs"] = "9ae0abe8f16a4e12f7f07ef454bba4131685e9dcfba33f7903fc01ad1bd7e96f",
        ["tests/pyannote/winograd-input-prototype/Driver.cs"] = "a37063df913a519995046035a44ffd8556f7465c423bfc639875abb81f7a2b27",
        ["tests/pyannote/winograd-output-blocks-prototype/Driver.cs"] = "f6ff64e5e669ba5534f29824e42c4fa9baea3c7dc73db263c0f7f3c1d7a8a4c7",
        ["tests/pyannote/winograd-prototype/Driver.cs"] = "a37063df913a519995046035a44ffd8556f7465c423bfc639875abb81f7a2b27",
        ["tests/pyannote/winograd-range-prototype/Driver.cs"] = "faca665f024df691bb7a7005478733400fbb683c764fd6676c1ee70ede648c7a",
        ["tests/pyannote/winograd-register-transform-prototype/Driver.cs"] = "f6ff64e5e669ba5534f29824e42c4fa9baea3c7dc73db263c0f7f3c1d7a8a4c7",
    };

    static string SourceHash(string code) => Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(
        System.Text.Encoding.UTF8.GetBytes(code.Replace("\r\n", "\n").TrimStart('\uFEFF'))));

    static bool IsArchivedSource(string relativePath, string code, IReadOnlyDictionary<string, string> archives) =>
        archives.TryGetValue(relativePath.Replace('\\', '/'), out string? expected) && SourceHash(code) == expected;

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
    public void ArchiveRecognitionRejectsChangedContentAndOtherPaths()
    {
        const string source = "void Historical(int value = 0) {}\n";
        var archives = new Dictionary<string, string> { ["archive/fixture.cs"] = SourceHash(source) };
        Assert.Single(FindOptionalParameters(source));
        Assert.True(IsArchivedSource("archive/fixture.cs", source, archives));
        Assert.True(IsArchivedSource("archive\\fixture.cs", source.Replace("\n", "\r\n"), archives));
        Assert.False(IsArchivedSource("src/fixture.cs", source, archives));
        Assert.False(IsArchivedSource("archive/fixture.cs", source + "void New(int added = 1) {}", archives));
    }

    [Fact]
    public void SourceTree_HasNoOptionalParameters()
    {
        string root = TestSupport.RepoRoot();
        var offenders = new List<string>();
        foreach (string file in TestSupport.SourceFiles("src", "tests"))
        {
            string relative = Path.GetRelativePath(root, file).Replace('\\', '/');
            string code = File.ReadAllText(file);
            if (ArchivedSources.ContainsKey(relative))
            {
                Assert.True(IsArchivedSource(relative, code, ArchivedSources),
                    "Historical experiment source changed; preserve it and use a successor: " + relative);
                continue;
            }
            var hits = FindOptionalParameters(code);
            foreach (string h in hits) offenders.Add(relative + ": " + h);
        }
        Assert.True(offenders.Count == 0, "Optional parameters found:\n" + string.Join("\n", offenders.Take(20)));
    }
}
