namespace Lokad.Onnx.Backend.Tests;

using System.Reflection;
using System.Runtime.CompilerServices;

public class AssemblyBoundaryTests
{
    [Fact]
    public void CoreHasNoParserDependencyOrGeneratedMessages()
    {
        var core = typeof(ComputationalGraph).Assembly;
        Assert.DoesNotContain(core.GetReferencedAssemblies(), reference => reference.Name == "Google.Protobuf" || reference.Name == "Lokad.Onnx.Protobuf");
        Assert.DoesNotContain(core.GetExportedTypes(), type => type.Namespace == "Onnx" || type == typeof(OnnxImport) || type == typeof(ProtoConversions));
        var adapter = typeof(OnnxImport).Assembly;
        Assert.Equal("Lokad.Onnx.Protobuf", adapter.GetName().Name);
        Assert.Same(adapter, typeof(global::Onnx.ModelProto).Assembly);
        Assert.Same(adapter, typeof(ProtoConversions).Assembly);
        Assert.Empty(Friends(adapter));
    }

    [Fact]
    public void OnlyBackendTestsCanAccessCoreInternals()
    {
        Assert.Equal(new[] { "Lokad.Onnx.Backend.Tests" }, Friends(typeof(ComputationalGraph).Assembly));
        Assert.Empty(Friends(typeof(ParakeetTranscriber).Assembly));
    }

    static string[] Friends(Assembly assembly) => assembly
        .GetCustomAttributes<InternalsVisibleToAttribute>()
        .Select(attribute => attribute.AssemblyName).ToArray();
}
