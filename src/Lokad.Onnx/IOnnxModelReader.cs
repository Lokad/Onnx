namespace Lokad.Onnx;

/// <summary>Caller-supplied parser producing the engine's neutral model description.</summary>
/// <remarks>
/// Return a fresh description owned by the caller on every read. Model.Load consumes
/// it without copying its collections or tensor payloads. A shared reader must allow
/// independent concurrent calls without sharing mutable model state.
/// </remarks>
public interface IOnnxModelReader
{
    /// <summary>Reads a model file, resolving external weights relative to its path.</summary>
    OnnxModel Read(string path);

    /// <summary>Reads borrowed model memory synchronously, without retaining it.</summary>
    /// <remarks>The returned description and payloads must remain valid after the
    /// caller reuses its input memory. Readers must explicitly reject unresolved
    /// external weights when no file context is available.</remarks>
    OnnxModel Read(ReadOnlyMemory<byte> data);
}
