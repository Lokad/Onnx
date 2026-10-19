using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

internal static class SelfTests
{
    internal static void Run()
    {
        string root = Path.Combine(AppContext.BaseDirectory, "Fixtures");
        MetadataReference[] references = ((string)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES")!).Split(Path.PathSeparator)
            .Select(path => MetadataReference.CreateFromFile(path)).ToArray();
        CompilationInput Fixture(string name)
        {
            string path = Path.Combine(root, name);
            var tree = CSharpSyntaxTree.ParseText(File.ReadAllText(path), path: path);
            return new CompilationInput(CSharpCompilation.Create("AuditFixture", [tree], references,
                new CSharpCompilationOptions(OutputKind.ConsoleApplication)), true);
        }
        Retention Reflection(string consumer) => new("AuditFixture:M:AuditFixtures.Program.Reflected", "reflection",
            "Explicit fixture reflection call", consumer, "\"Reflected\"");
        var unused = SourceAudit.Analyze(root, [Fixture("Unused.cs.txt")], [Reflection("Unused.cs.txt")]);
        Require(unused.RetentionErrors.Count == 0 && unused.Unreviewed.Select(member => member.MemberId)
            .SequenceEqual(new[] { "AuditFixture:M:AuditFixtures.Program.UnusedHelper", "AuditFixture:P:AuditFixtures.Indexed.Item(System.Int32)" }), "unused private helper and unconsumed internal indexer must be detected");
        Require(unused.Members.Single(member => member.MemberId == "AuditFixture:M:AuditFixtures.ExternalApi.CalledOutsideThisRepository").Classification == "public-api",
            "external public API must remain a root");

        var retained = SourceAudit.Analyze(root, [Fixture("Retained.cs.txt")], [Reflection("Retained.cs.txt")]);
        Require(retained.Unreviewed.Count == 0 && retained.RetentionErrors.Count == 0, "entrypoint, constructors, disposal, interface and reflection consumers must remain retained");
        Require(retained.Members.Single(member => member.MemberId == "AuditFixture:M:AuditFixtures.Policy.Convert(System.Single)").Classification == "interface implementation",
            "implicit static-abstract interface implementation must remain retained");
        var diagnostic = Fixture("Diagnostic.cs.txt");
        string consumerPath = Path.Combine(root, "DiagnosticConsumer.cs.txt");
        var testCompilation = CSharpCompilation.Create("AuditTests",
            [CSharpSyntaxTree.ParseText(File.ReadAllText(consumerPath), path: consumerPath)],
            references.Append(diagnostic.Compilation.ToMetadataReference()), new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        var testOnlyInputs = new[] { diagnostic, new CompilationInput(testCompilation, false) };
        const string diagnosticId = "AuditFixture:M:AuditFixtures.Diagnostic.Read";
        var testOnly = SourceAudit.Analyze(root, testOnlyInputs, []);
        Require(testOnly.Unreviewed.Single().MemberId == diagnosticId && testOnly.Unreviewed.Single().Classification == "test-only",
            "test-only consumers must be distinguished from production callers");
        var reviewedDiagnostic = new Retention(diagnosticId, "test-only", "Intentional test diagnostic", "DiagnosticConsumer.cs.txt", null);
        var reviewedTestOnly = SourceAudit.Analyze(root, testOnlyInputs, [reviewedDiagnostic]);
        Require(reviewedTestOnly.Unreviewed.Count == 0 && reviewedTestOnly.RetentionErrors.Count == 0, "reviewed test diagnostics must remain accepted");
        var staleTestConsumer = SourceAudit.Analyze(root, testOnlyInputs, [reviewedDiagnostic with { Consumer = "absent.cs.txt" }]);
        Require(staleTestConsumer.RetentionErrors.Count == 1, "stale named test consumers must fail");
        var unregistered = SourceAudit.Analyze(root, [Fixture("Retained.cs.txt")], []);
        Require(unregistered.Unreviewed.Single().MemberId == "AuditFixture:M:AuditFixtures.Program.Reflected",
            "reflection strings must need explicit registration");
        var missingConsumer = SourceAudit.Analyze(root, [Fixture("Retained.cs.txt")], [Reflection("absent.cs.txt")]);
        Require(missingConsumer.RetentionErrors.Count == 1, "missing reflection consumer must fail");
        var stale = SourceAudit.Analyze(root, [Fixture("Unused.cs.txt")], [Reflection("Unused.cs.txt"),
            new Retention("AuditFixture:M:AuditFixtures.ExternalApi.CalledOutsideThisRepository", "retained", "Stale public API exception", null, null),
            new Retention("AuditFixture:M:AuditFixtures.Deleted", "retained", "Stale removed member", null, null)]);
        Require(stale.RetentionErrors.Count == 2, "stale member and unnecessary public-root exceptions must fail");
        var invalid = CSharpCompilation.Create("InvalidFixture", [CSharpSyntaxTree.ParseText("class Broken { UnknownType field; }")], references);
        try
        {
            SourceAudit.Analyze(root, [new CompilationInput(invalid, true)], []);
            throw new Exception("A compilation failure was reported clean.");
        }
        catch (InvalidOperationException) { }
        Console.WriteLine("PASS source-audit fixtures: unused detection; public/entrypoint/constructor/disposal/interface/static-interface/reflection roots; stale registrations; compilation failure.");
    }

    static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException("Fixture failure: " + message);
    }
}
