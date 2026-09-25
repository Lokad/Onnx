internal static class DisassemblyPolicy
{
    internal static bool Allowed(Dictionary<string, string?> flags)
    {
        return flags.Count == 2
            && flags.TryGetValue("DOTNET_JitDisasm", out string? methods)
            && methods == "RunFloatMatMulKernel RunBatchedFloatMatMul"
            && flags.TryGetValue("DOTNET_JitDisasmWithCodeBytes", out string? bytes)
            && bytes == "1";
    }
}
