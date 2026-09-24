global using Xunit;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

[assembly: CollectionBehavior(DisableTestParallelization = true)]

namespace Lokad.Onnx.Backend.Tests;

[CollectionDefinition("SequentialLogSink", DisableParallelization = true)]
public class SequentialLogSinkCollection { }

public class IdentityTests
{
    [Fact]
    public void ConsumedProductsAndInstructionModeMatch()
    {
        static string Sha(string path)
        {
            using var stream = File.OpenRead(path);
            return Convert.ToHexStringLower(SHA256.HashData(stream));
        }
        var core = typeof(ComputationalGraph).Assembly.Location;
        var data = typeof(ParakeetTranscriber).Assembly.Location;
        Assert.Equal(Environment.GetEnvironmentVariable("PACKING_CORE_SHA"), Sha(core));
        Assert.Equal(Environment.GetEnvironmentVariable("PACKING_DATA_SHA"), Sha(data));
        Assert.Equal(Environment.GetEnvironmentVariable("PACKING_RUNTIME"), Path.GetDirectoryName(core));
        Assert.Equal(Path.GetDirectoryName(core), Path.GetDirectoryName(data));
        Assert.Equal(Environment.GetEnvironmentVariable("PACKING_AVX512") == "1", Avx512F.IsSupported);
        Assert.True(Avx2.IsSupported && Fma.IsSupported);
        using var process = Process.GetCurrentProcess();
        Assert.Equal(4L, process.ProcessorAffinity.ToInt64());
        Assert.Equal(1, Environment.ProcessorCount);
        Assert.Equal(".NET 10.0.8", RuntimeInformation.FrameworkDescription);
        Assert.DoesNotContain(process.Modules.Cast<ProcessModule>(), m => m.FileName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase));
        using var output = new FileStream(Environment.GetEnvironmentVariable("PACKING_IDENTITY_FILE")!, FileMode.CreateNew, FileAccess.Write);
        JsonSerializer.Serialize(output, new { passed = true, pid = process.Id,
            core_path = core, data_path = data, core_sha256 = Sha(core), data_sha256 = Sha(data),
            consumer_sha256 = Sha(Assembly.GetExecutingAssembly().Location),
            avx512 = Avx512F.IsSupported, avx2 = Avx2.IsSupported, fma = Fma.IsSupported,
            affinity = process.ProcessorAffinity.ToInt64(), processor_count = Environment.ProcessorCount,
            runtime = RuntimeInformation.FrameworkDescription });
    }
}
