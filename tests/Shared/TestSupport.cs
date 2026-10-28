using System.Reflection;

namespace Lokad.Onnx.Tests.Support;

/// <summary>
/// Test-only discovery shared by both test assemblies through linking (ships nowhere):
/// repository-root discovery, CLI build and committed-asset paths.
/// Numerical expectations stay in the test classes; this file only finds files.
/// </summary>
internal static class TestSupport
{
    internal static string RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Lokad.Onnx.slnx"))) return dir.FullName;
            dir = dir.Parent;
        }
        throw new DirectoryNotFoundException("Repository root with Lokad.Onnx.slnx not found.");
    }

    internal static string CommittedModel(string fileName) =>
        Path.Combine(RepoRoot(), "tests", "Lokad.Onnx.Backend.Tests", "models", fileName);

    internal static string CommittedImage(string fileName) =>
        Path.Combine(RepoRoot(), "tests", "Lokad.Onnx.Backend.Tests", "images", fileName);

    internal static string CliDll()
    {
        string configuration = typeof(TestSupport).Assembly.GetCustomAttribute<AssemblyConfigurationAttribute>()?.Configuration
            ?? throw new InvalidOperationException("Cannot determine the test build configuration.");
        string framework = typeof(TestSupport).Assembly.GetCustomAttribute<System.Runtime.Versioning.TargetFrameworkAttribute>()?.FrameworkName
            ?? throw new InvalidOperationException("Cannot determine the test target framework.");
        var version = new System.Runtime.Versioning.FrameworkName(framework).Version;
        return Path.Combine(RepoRoot(), "src", "Lokad.Onnx.CLI", "bin", configuration,
            $"net{version.Major}.{version.Minor}", "Lokad.Onnx.CLI.dll");
    }
}
