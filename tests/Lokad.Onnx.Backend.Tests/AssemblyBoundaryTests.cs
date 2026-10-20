namespace Lokad.Onnx.Backend.Tests;

using System.Reflection;
using System.Runtime.CompilerServices;

public class AssemblyBoundaryTests
{
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
