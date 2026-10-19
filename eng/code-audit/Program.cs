using System.Collections.Concurrent;
using System.Text.Json;
using Microsoft.Build.Locator;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.MSBuild;

try
{
    if (args.SequenceEqual(new[] { "--self-test" }))
    {
        SelfTests.Run();
        return 0;
    }
    string? repo = null, output = null;
    bool check = false;
    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--repo" when i + 1 < args.Length: repo = Path.GetFullPath(args[++i]); break;
            case "--output" when i + 1 < args.Length: output = Path.GetFullPath(args[++i]); break;
            case "--check": check = true; break;
            default: throw new ArgumentException("Usage: CodeAudit --self-test | --repo <directory> [--check] --output <directory>");
        }
    }
    if (repo is null || output is null) throw new ArgumentException("Both --repo and --output are required.");
    MSBuildLocator.RegisterDefaults();
    var report = await InspectSolution(repo);
    Directory.CreateDirectory(output);
    File.WriteAllText(Path.Combine(output, "report.json"), JsonSerializer.Serialize(report, new JsonSerializerOptions
    {
        WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    }) + "\n");
    foreach (var finding in report.Unreviewed)
        Console.WriteLine($"UNREVIEWED {finding.MemberId} ({finding.Classification}) at {finding.File}:{finding.Line}");
    foreach (string error in report.RetentionErrors) Console.WriteLine("RETENTION " + error);
    Console.WriteLine($"Audited {report.Projects} compilations and {report.Members.Count} declarations: {report.Unreviewed.Count} unreviewed; {report.RetentionErrors.Count} retention errors.");
    if (check && (report.Unreviewed.Count != 0 || report.RetentionErrors.Count != 0)) return 1;
    Console.WriteLine(check ? "PASS source audit" : "Report written; use --check to enforce reviewed findings.");
    return 0;
}
catch (Exception error)
{
    Console.Error.WriteLine("FAIL source audit: " + error);
    return 2;
}

static async Task<AuditReport> InspectSolution(string repo)
{
    using var workspace = MSBuildWorkspace.Create(new Dictionary<string, string>
    {
        ["Configuration"] = "Release", ["NuGetAudit"] = "false",
        // ResolvePackageAssets can replay advisory warnings from an earlier
        // restore as workspace failures. They do not change C# compilation.
        ["NoWarn"] = "NU1901;NU1902;NU1903;NU1904"
    });
    var failures = new ConcurrentQueue<string>();
#pragma warning disable CS0618 // Supported by all SDK Roslyn versions selected through global.json.
    workspace.WorkspaceFailed += (_, e) =>
    {
        if (e.Diagnostic.Kind == WorkspaceDiagnosticKind.Failure) failures.Enqueue(e.Diagnostic.Message);
        else Console.WriteLine("Workspace: " + e.Diagnostic.Message);
    };
#pragma warning restore CS0618
    var solution = await workspace.OpenSolutionAsync(Path.Combine(repo, "Lokad.Onnx.slnx"));
    int expectedProjects = System.Xml.Linq.XDocument.Load(Path.Combine(repo, "Lokad.Onnx.slnx")).Descendants("Project").Count();
    if (solution.Projects.Count() != expectedProjects)
        throw new InvalidOperationException($"Workspace loaded {solution.Projects.Count()} of {expectedProjects} solution projects.\n" + string.Join("\n", failures.Distinct()));
    var inputs = new List<CompilationInput>();
    foreach (var project in solution.Projects)
    {
        var compilation = await project.GetCompilationAsync() ?? throw new InvalidOperationException("Cannot compile " + project.Name);
        string path = Path.GetRelativePath(repo, project.FilePath!).Replace('\\', '/');
        inputs.Add(new CompilationInput(compilation, path.StartsWith("src/", StringComparison.Ordinal)));
        Console.WriteLine("Loaded " + project.Name);
    }
    if (failures.Count != 0) throw new InvalidOperationException("Workspace load failed: " + string.Join("\n", failures.Distinct()));
    var retained = JsonSerializer.Deserialize<Retention[]>(File.ReadAllText(Path.Combine(repo, "eng/code-audit/retained-members.json")),
        new JsonSerializerOptions { PropertyNameCaseInsensitive = true }) ?? throw new InvalidDataException("Invalid retained-members.json");
    return SourceAudit.Analyze(repo, inputs, retained);
}
