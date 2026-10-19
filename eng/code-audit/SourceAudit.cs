using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

internal sealed record CompilationInput(Compilation Compilation, bool Product);
internal sealed record Retention(string MemberId, string Category, string Reason, string? Consumer, string? Needle);
internal sealed record ReferenceUse(string File, int Line, bool Product);
internal sealed record MemberFinding(string MemberId, string Name, string File, int Line, string Classification,
    string? RetentionReason, IReadOnlyList<ReferenceUse> Uses);
internal sealed record AuditReport(int Projects, IReadOnlyList<MemberFinding> Members,
    IReadOnlyList<MemberFinding> Unreviewed, IReadOnlyList<string> RetentionErrors);

internal static class SourceAudit
{
    internal static string? Key(ISymbol? symbol)
    {
        if (symbol is IMethodSymbol { ReducedFrom: not null } reduced) symbol = reduced.ReducedFrom;
        if (symbol is IMethodSymbol { AssociatedSymbol: not null } accessor) symbol = accessor.AssociatedSymbol;
        symbol = symbol?.OriginalDefinition;
        return symbol?.ContainingAssembly?.Name is string assembly && symbol.GetDocumentationCommentId() is string id ? assembly + ":" + id : null;
    }

    static bool PublicApi(ISymbol symbol)
    {
        for (ISymbol? current = symbol; current is not null && current is not INamespaceSymbol; current = current.ContainingType)
            if (current.DeclaredAccessibility is not (Accessibility.Public or Accessibility.Protected or Accessibility.ProtectedOrInternal)) return false;
        return true;
    }

    static string? FrameworkRoot(ISymbol symbol, IMethodSymbol? entrypoint)
    {
        if (symbol is IMethodSymbol method)
        {
            if (Key(method) == Key(entrypoint) && entrypoint is not null) return "entrypoint";
            if (method.MethodKind is MethodKind.StaticConstructor or MethodKind.Destructor) return "runtime lifecycle";
            if (method.IsOverride || method.IsAbstract || method.ExplicitInterfaceImplementations.Length != 0) return "virtual/interface contract";
        }
        if (symbol is IPropertySymbol property && (property.IsOverride || property.IsAbstract || property.ExplicitInterfaceImplementations.Length != 0))
            return "virtual/interface contract";
        if (symbol is IEventSymbol eventSymbol && (eventSymbol.IsOverride || eventSymbol.IsAbstract || eventSymbol.ExplicitInterfaceImplementations.Length != 0))
            return "virtual/interface contract";
        if (symbol is INamedTypeSymbol type && entrypoint is not null)
            for (var current = entrypoint.ContainingType; current is not null; current = current.ContainingType)
                if (SymbolEqualityComparer.Default.Equals(current.OriginalDefinition, type.OriginalDefinition)) return "entrypoint containing type";
        if (symbol.ContainingType is { } containing && symbol is IMethodSymbol or IPropertySymbol or IEventSymbol)
            foreach (var contract in containing.AllInterfaces)
                foreach (var member in contract.GetMembers())
                    if (containing.FindImplementationForInterfaceMember(member) is { } implementation &&
                        SymbolEqualityComparer.Default.Equals(implementation.OriginalDefinition, symbol.OriginalDefinition))
                        return "interface implementation";
        if (symbol.GetAttributes().Any(attribute => attribute.AttributeClass?.ToDisplayString() is "CommandLine.OptionAttribute" or "CommandLine.VerbAttribute"))
            return "CommandLine attribute binding";
        return null;
    }

    internal static AuditReport Analyze(string repo, IReadOnlyList<CompilationInput> inputs, IReadOnlyList<Retention> retained)
    {
        repo = Path.GetFullPath(repo);
        string Relative(string path) => Path.GetRelativePath(repo, path).Replace('\\', '/');
        var declarations = new Dictionary<string, (ISymbol Symbol, string File, int Line, string? Root)>();
        var uses = new Dictionary<string, HashSet<ReferenceUse>>();
        foreach (var input in inputs)
        {
            if (input.Compilation.Language != LanguageNames.CSharp) throw new InvalidOperationException("Unsupported audit language: " + input.Compilation.Language);
            var errors = input.Compilation.GetDiagnostics().Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error).ToArray();
            if (errors.Length != 0) throw new InvalidOperationException($"{input.Compilation.AssemblyName}: {errors.Length} compilation errors\n" + string.Join("\n", errors.Take(8).Select(error => error.ToString())));
            var entrypoint = input.Compilation.GetEntryPoint(default);
            foreach (var tree in input.Compilation.SyntaxTrees)
            {
                var root = tree.GetRoot();
                var model = input.Compilation.GetSemanticModel(tree);
                string path = Relative(tree.FilePath);
                bool handwritten = !path.Contains("/obj/") && !path.Contains("/bin/") && !path.Contains("/Generated/") && !path.EndsWith(".g.cs", StringComparison.OrdinalIgnoreCase);
                if (input.Product && handwritten)
                    foreach (var node in root.DescendantNodes())
                    {
                        if (node is not (BaseMethodDeclarationSyntax or PropertyDeclarationSyntax or IndexerDeclarationSyntax or BaseTypeDeclarationSyntax or DelegateDeclarationSyntax or EnumMemberDeclarationSyntax or EventDeclarationSyntax)
                            && node is not VariableDeclaratorSyntax { Parent.Parent: FieldDeclarationSyntax or EventFieldDeclarationSyntax }) continue;
                        var symbol = model.GetDeclaredSymbol(node);
                        if (symbol is not (IMethodSymbol or IPropertySymbol or IFieldSymbol or INamedTypeSymbol or IEventSymbol) || Key(symbol) is not string key) continue;
                        declarations[key] = (symbol, path, tree.GetLineSpan(node.Span).StartLinePosition.Line + 1,
                            PublicApi(symbol) ? "public-api" : FrameworkRoot(symbol, entrypoint));
                    }
                // Generated callers count even though generated declarations are not findings.
                foreach (var node in root.DescendantNodes())
                {
                    if (node is not (SimpleNameSyntax or ObjectCreationExpressionSyntax or ImplicitObjectCreationExpressionSyntax
                        or InvocationExpressionSyntax or BinaryExpressionSyntax or PrefixUnaryExpressionSyntax or PostfixUnaryExpressionSyntax
                        or CastExpressionSyntax or AttributeSyntax or ConstructorInitializerSyntax or ElementAccessExpressionSyntax)) continue;
                    var info = model.GetSymbolInfo(node);
                    // Unresolved/dynamic candidate symbols are not proven references.
                    if (Key(info.Symbol) is not string target) continue;
                    var ownerNode = node.Ancestors().FirstOrDefault(parent => parent is BaseMethodDeclarationSyntax or PropertyDeclarationSyntax or IndexerDeclarationSyntax or VariableDeclaratorSyntax);
                    if (ownerNode is not null && Key(model.GetDeclaredSymbol(ownerNode)) == target) continue;
                    if (!uses.TryGetValue(target, out var references)) uses[target] = references = [];
                    references.Add(new ReferenceUse(path, tree.GetLineSpan(node.Span).StartLinePosition.Line + 1, input.Product));
                }
            }
        }

        var retainedById = new Dictionary<string, Retention>();
        var retentionErrors = new List<string>();
        foreach (var item in retained)
        {
            if (string.IsNullOrWhiteSpace(item.MemberId) || !retainedById.TryAdd(item.MemberId, item))
                retentionErrors.Add("Missing/duplicate member identity: " + item.MemberId);
            if (string.IsNullOrWhiteSpace(item.Reason)) retentionErrors.Add("Missing reason: " + item.MemberId);
            if (item.Category is not ("test-only" or "reflection" or "retained")) retentionErrors.Add("Unknown category: " + item.MemberId);
            if (!declarations.ContainsKey(item.MemberId)) retentionErrors.Add("Stale member no longer exists: " + item.MemberId);
        }
        var findings = new List<MemberFinding>();
        var unreviewed = new List<MemberFinding>();
        foreach (var (key, declaration) in declarations.OrderBy(pair => pair.Key, StringComparer.Ordinal))
        {
            uses.TryGetValue(key, out var references);
            references ??= [];
            string category = declaration.Root ?? (references.Any(reference => reference.Product) ? "production" : references.Count > 0 ? "test-only" : "unreferenced");
            string? reason = null;
            bool candidate = category is "test-only" or "unreferenced";
            if (retainedById.TryGetValue(key, out var retention))
            {
                reason = retention.Reason;
                if (!candidate) retentionErrors.Add("Stale exception now retained by " + category + ": " + key);
                if (retention.Category == "test-only" && (category != "test-only" ||
                    retention.Consumer is not null && !references.Any(reference => !reference.Product && reference.File == retention.Consumer)))
                    retentionErrors.Add("Test consumer missing: " + key);
                if (retention.Category == "reflection")
                {
                    bool validConsumer = false;
                    if (retention.Consumer is not null && retention.Needle is not null)
                    {
                        string consumer = Path.GetFullPath(Path.Combine(repo, retention.Consumer));
                        string relative = Relative(consumer);
                        validConsumer = !relative.StartsWith("../", StringComparison.Ordinal) && !Path.IsPathRooted(relative)
                            && File.Exists(consumer) && File.ReadAllText(consumer).Contains(retention.Needle, StringComparison.Ordinal);
                    }
                    if (!validConsumer) retentionErrors.Add("Registered reflection consumer missing: " + key);
                }
                category = retention.Category;
            }
            var finding = new MemberFinding(key, declaration.Symbol.ToDisplayString(), declaration.File, declaration.Line, category, reason,
                references.OrderBy(reference => reference.File, StringComparer.Ordinal).ThenBy(reference => reference.Line).ToArray());
            findings.Add(finding);
            if (candidate && !retainedById.ContainsKey(key)) unreviewed.Add(finding);
        }
        return new AuditReport(inputs.Count, findings, unreviewed, retentionErrors);
    }
}
