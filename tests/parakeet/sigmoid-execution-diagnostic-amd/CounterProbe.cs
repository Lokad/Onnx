using System.Diagnostics;

internal static class CounterProbe
{
    internal readonly record struct Snapshot(long Allocated, int Gen0, int Gen1, int Gen2);
    internal sealed record Observation(int iteration, long allocated, int gen0, int gen1, int gen2);

    internal static Snapshot Start() => new(GC.GetAllocatedBytesForCurrentThread(),
        GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2));

    internal static Observation End(Snapshot before, int iteration)
    {
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before.Allocated;
        return new(iteration, allocated, GC.CollectionCount(0) - before.Gen0,
            GC.CollectionCount(1) - before.Gen1, GC.CollectionCount(2) - before.Gen2);
    }

    internal static bool Allowed(Dictionary<string, string> flags) => flags.Count == 2
        && flags.TryGetValue("DOTNET_JitDisasm", out string? methods)
        && methods == "Lokad.Onnx.CPUExecutionProvider:Sigmoid Lokad.Onnx.MathOps:ExpVector"
        && flags.TryGetValue("DOTNET_JitDisasmWithCodeBytes", out string? bytes) && bytes == "1";
}
