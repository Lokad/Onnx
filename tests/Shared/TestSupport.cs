namespace Lokad.Onnx.Tests.Support;

/// <summary>
/// Test-only discovery shared by both test assemblies through linking (ships nowhere):
/// repository-root discovery and committed-asset paths.
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
}
