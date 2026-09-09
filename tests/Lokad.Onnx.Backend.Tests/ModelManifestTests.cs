using System.Security.Cryptography;
using System.Text.Json;
using Xunit.Abstractions;

namespace Lokad.Onnx.Backend.Tests;

using Lokad.Onnx.Tests.Support;

/// <summary>
/// Guards the T04 manifest rule: every model oracle asset is identified by a
/// recorded hash and size, and present assets must match. Absent assets skip
/// unless the local lane is explicitly required, in which case they fail.
/// </summary>
public class ModelManifestTests
{
    readonly ITestOutputHelper output;

    public ModelManifestTests(ITestOutputHelper output)
    {
        this.output = output;
    }

    static string ManifestPath() =>
        Path.Combine(TestSupport.RepoRoot(), "tests", "Lokad.Onnx.Backend.Tests", "ModelManifest.json");

    [Fact]
    public void Manifest_IdentifiesPresentAssets()
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(ManifestPath()));
        var missing = new List<string>();
        int verified = 0;
        foreach (var entry in doc.RootElement.GetProperty("models").EnumerateArray())
        {
            string key = entry.GetProperty("key").GetString()!;
            string file = entry.GetProperty("file").GetString()!;
            string path = ModelFixture.FindModelPath(file.Split('/')) ?? string.Empty;
            if (string.IsNullOrEmpty(path)) { missing.Add(key + " (" + file + ")"); continue; }
            string expectedHash = entry.GetProperty("sha256").GetString()!;
            long expectedBytes = entry.GetProperty("bytes").GetInt64();
            using var sha = SHA256.Create();
            using var stream = File.OpenRead(path);
            string actualHash = Convert.ToHexString(sha.ComputeHash(stream));
            Assert.True(string.Equals(actualHash, expectedHash, StringComparison.OrdinalIgnoreCase),
                key + ": sha256 mismatch (asset drift?).");
            Assert.Equal(expectedBytes, new FileInfo(path).Length);
            verified++;
            output.WriteLine(key + " " + file + " sha256=" + actualHash.Substring(0, 12)
                + " ort=" + doc.RootElement.GetProperty("ortOracle").GetString()
                + " generator=" + entry.GetProperty("generator").GetString());
        }
        if (missing.Count > 0 && Environment.GetEnvironmentVariable(ModelFixture.LaneVariable) == "1")
        {
            Assert.Fail("Manifest assets requested via lane but missing: " + string.Join("; ", missing));
        }
        Skip.If(verified == 0, "No manifest assets present; set " + ModelFixture.LaneVariable + "=1 to require them.");
    }
}
