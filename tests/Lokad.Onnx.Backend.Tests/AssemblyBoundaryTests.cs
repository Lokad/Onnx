namespace Lokad.Onnx.Backend.Tests;

using System.Reflection;
using System.Runtime.CompilerServices;
using Lokad.Onnx.Tests.Support;

public class AssemblyBoundaryTests
{
    [Fact]
    public void EngineAndCompanionsUseTheTestBuildConfiguration()
    {
        var expected = typeof(AssemblyBoundaryTests).Assembly.GetCustomAttribute<AssemblyConfigurationAttribute>()!.Configuration;
        foreach (var assembly in new[] { typeof(Model).Assembly, typeof(OnnxImport).Assembly, typeof(ParakeetTranscriber).Assembly,
            Assembly.LoadFrom(TestSupport.CliDll()) })
            Assert.Equal(expected, assembly.GetCustomAttribute<AssemblyConfigurationAttribute>()!.Configuration);
    }

    [Fact]
    public void CoreOwnsTheIntegratedParserAndGeneratedMessages()
    {
        var core = typeof(ComputationalGraph).Assembly;
        Assert.Contains(core.GetReferencedAssemblies(), reference => reference.Name == "Google.Protobuf");
        Assert.DoesNotContain(core.GetReferencedAssemblies(), reference => reference.Name == "Lokad.Onnx.Protobuf");
        Assert.Same(core, typeof(OnnxImport).Assembly);
        Assert.Same(core, typeof(global::Onnx.ModelProto).Assembly);
        Assert.Same(core, typeof(ProtoConversions).Assembly);
        Assert.Null(core.GetType("Lokad.Onnx.IOnnxModelReader"));
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
