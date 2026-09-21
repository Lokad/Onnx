using System;
using System.Reflection;
using System.Runtime.InteropServices;

namespace Lokad.Onnx.Backend.Tests;

public class LstmPanelOverflowRefusalTests
{
    delegate IDisposable? CreatePanels(ReadOnlySpan<float> w, ReadOnlySpan<float> r,
        int inputSize, int hiddenSize, int directions, int sequence, TensorExecutionOptions options);

    [Fact]
    public void UnrepresentableOptionalPanelDeclinesBeforeAllocationOrWeightReads()
    {
        var type = typeof(CPUExecutionProvider).GetNestedType("LstmProjectionPanels", BindingFlags.NonPublic)!;
        var method = type.GetMethod("Create", BindingFlags.Static | BindingFlags.NonPublic)!;
        var create = method.CreateDelegate<CreatePanels>();
        float sentinel = 12345f;
        // These spans only exercise admission. Neither the original checked
        // addition nor its replacement may reach an allocation or a weight read.
        // The original source throws OverflowException on the combined length.
        var w = MemoryMarshal.CreateReadOnlySpan(ref sentinel, int.MaxValue - 64);
        var r = MemoryMarshal.CreateReadOnlySpan(ref sentinel, 128);
        using var panels = create(w, r, 1, 16, 1, 8, TensorExecutionOptions.Auto);
        Assert.Null(panels);
        Assert.Equal(12345f, sentinel);
    }
}
