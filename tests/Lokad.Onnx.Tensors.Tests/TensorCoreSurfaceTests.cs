using System.Reflection;
using System.Security.Cryptography;
using System.Text;

namespace Lokad.Onnx.Tensors.Tests;

// Freeze compiled signatures, allowing implementation bodies and formatting to change.
public class TensorCoreSurfaceTests
{
    [Fact]
    public void PortedCoreSurface_MatchesInventory()
    {
        var members = new[] { typeof(Tensor<>), typeof(DenseTensor<>) }
            .SelectMany(type => type.GetMembers(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly)
                .Select(member => type.Name + ":" + member))
            .OrderBy(line => line, StringComparer.Ordinal);
        var inventory = string.Join('\n', members);
        var digest = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(inventory)));
        // Intentional API changes require reviewing the signatures before updating this baseline.
        Assert.True(digest == "1DD5104C2F40DCD50A3E8FA0D831F111B84D1D31D785D2CDA7DDB62BF31A38A1",
            "Compiled public API changed. Actual signatures:\n" + inventory);
        Assert.True(typeof(Tensor<>).IsAbstract);
        Assert.Equal(typeof(Tensor<>), typeof(DenseTensor<>).BaseType!.GetGenericTypeDefinition());
    }
}
