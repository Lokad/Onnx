using System;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;

internal static class KernelAccess
{
    delegate float[] PrepareCall(ReadOnlySpan<float> weights, int c, int m, int lanes);
    delegate bool PlanCall(int c, int m, int h, int w, out int ni, out int np, out int no);
    delegate bool DirectCall(ReadOnlySpan<float> input, ReadOnlySpan<float> prepared,
        ReadOnlySpan<float> bias, ReadOnlySpan<float> residual, Span<float> output,
        Span<float> packedInput, Span<float> packedOutput, int c, int m, int h, int w,
        int stride, int lanes, bool relu);
    delegate bool WinogradCall(ReadOnlySpan<float> input, ReadOnlySpan<float> prepared,
        ReadOnlySpan<float> bias, ReadOnlySpan<float> residual, Span<float> output,
        Span<float> transformed, Span<float> products, Span<float> blocked,
        int c, int m, int h, int w, int lanes, bool relu);
    delegate bool RangeCall(ReadOnlySpan<float> input);

    static PrepareCall prepare = null!, prepareWinograd = null!;
    static PlanCall plan = null!;
    static DirectCall direct = null!;
    static WinogradCall winograd = null!;
    static RangeCall range = null!;
    internal static string CoreHash { get; private set; } = "";

    internal static void Bind(string path)
    {
        if (CoreHash.Length != 0) throw new InvalidOperationException("Already bound");
        path = Path.GetFullPath(path);
        CoreHash = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
        var assembly = Assembly.LoadFrom(path);
        if (assembly.Location != path) throw new InvalidOperationException("Unexpected loaded core");
        var type = assembly.GetType("Lokad.Onnx.ConvBlockedSpatial", throwOnError: true)!;
        T Method<T>(string name) where T : Delegate => type.GetMethod(name,
            BindingFlags.Static | BindingFlags.NonPublic)!.CreateDelegate<T>();
        prepare = Method<PrepareCall>("Prepare");
        prepareWinograd = Method<PrepareCall>("PrepareWinograd");
        plan = Method<PlanCall>("PlanWinograd");
        direct = Method<DirectCall>("Execute");
        winograd = Method<WinogradCall>("ExecuteWinograd");
        range = Method<RangeCall>("EpilogueRange");
    }

    internal static float[] Prepare(ReadOnlySpan<float> weights, int c, int m, int lanes)
        => prepare(weights,c,m,lanes);
    internal static float[]? PrepareWinograd(ReadOnlySpan<float> weights, int c, int m, int lanes)
        => prepareWinograd(weights,c,m,lanes);
    internal static bool PlanWinograd(int c, int m, int h, int w, out int ni, out int np, out int no)
        => plan(c,m,h,w,out ni,out np,out no);
    internal static bool Execute(ReadOnlySpan<float> input, ReadOnlySpan<float> prepared,
        ReadOnlySpan<float> bias, ReadOnlySpan<float> residual, Span<float> output,
        Span<float> packedInput, Span<float> packedOutput, int c, int m, int h, int w,
        int stride, int lanes, bool relu)
        => direct(input,prepared,bias,residual,output,packedInput,packedOutput,c,m,h,w,stride,lanes,relu);
    internal static bool ExecuteWinograd(ReadOnlySpan<float> input, ReadOnlySpan<float> prepared,
        ReadOnlySpan<float> bias, ReadOnlySpan<float> residual, Span<float> output,
        Span<float> transformed, Span<float> products, Span<float> blocked,
        int c, int m, int h, int w, int lanes, bool relu)
        => winograd(input,prepared,bias,residual,output,transformed,products,blocked,c,m,h,w,lanes,relu);
    static bool EpilogueRange(ReadOnlySpan<float> input) => range(input);
}
