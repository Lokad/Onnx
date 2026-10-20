using Lokad.Onnx.Tests.Support;

namespace Lokad.Onnx.Backend.Tests;

[Collection("SequentialLogSink")]
public class ImportErrorOwnershipTests
{
    [Fact]
    public async Task ConcurrentThrowingLoads_OwnTheirErrors()
    {
        var errors = await Task.WhenAll(Enumerable.Range(0, 8).Select(i => Task.Run(() =>
        {
            string path = Path.Combine(Path.GetTempPath(), $"missing-onnx-{Guid.NewGuid():N}-{i}.onnx");
            var error = Assert.Throws<InvalidDataException>(() => OnnxImport.LoadOrThrow(path));
            return (path, error);
        })));
        Assert.NotNull(OnnxImport.Load(TestSupport.CommittedModel("mnist-8.onnx")));
        foreach (var (path, error) in errors)
        {
            Assert.Contains(path, error.Message);
            Assert.Equal(path, Assert.IsType<FileNotFoundException>(error.InnerException).FileName);
        }
    }

    [Fact]
    public void ThrowingLoad_DoesNotOverwriteCompatibilityDiagnostics()
    {
        Assert.Null(OnnxImport.Load(new byte[] { 0xff }));
        var previousCause = OnnxImport.LastErrorCause;
        string? previousMessage = OnnxImport.LastErrorMessage;
        Assert.NotNull(OnnxImport.LoadOrThrow(TestSupport.CommittedModel("mnist-8.onnx")));
        Assert.Throws<InvalidDataException>(() => OnnxImport.LoadOrThrow("missing-" + Guid.NewGuid() + ".onnx"));
        Assert.Same(previousCause, OnnxImport.LastErrorCause);
        Assert.Equal(previousMessage, OnnxImport.LastErrorMessage);
    }

    [Fact]
    public void NegativeBudget_IsReportedAsThisCallsCause()
    {
        var error = Assert.Throws<InvalidDataException>(() => OnnxImport.LoadOrThrow("unused.onnx", -1));
        Assert.IsType<ArgumentOutOfRangeException>(error.InnerException);
        Assert.Contains("unused.onnx", error.Message);
    }
}
