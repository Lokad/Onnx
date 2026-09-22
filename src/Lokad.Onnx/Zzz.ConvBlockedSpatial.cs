using System;
using System.Buffers;
using System.Linq;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Lokad.Onnx;

// Isolated component. No graph/provider dispatch or shared product changes.
namespace Lokad.Onnx;

internal static unsafe partial class ConvBlockedSpatial
{
    internal static int Output(int size, int stride) => checked((size + stride - 1) / stride);

    internal static void Geometry(int c, int m, int h, int w, int stride, int lanes)
    {
        if (lanes is not (8 or 16) || c < 16 || c % 16 != 0 || m < 32 || m % 16 != 0
            || h < 1 || w < 1 || stride is not (1 or 2)) throw new ArgumentException("Unsupported geometry");
        _ = checked(c * checked((h + 2) * (w + 2)));
        _ = checked(m * Output(h, stride) * Output(w, stride));
        _ = checked(((m + 2 * lanes - 1) / (2 * lanes)) * (2 * lanes) * c * 9);
    }

    internal static float[] Prepare(ReadOnlySpan<float> weights, int c, int m, int lanes)
    {
        Geometry(c, m, 1, 1, 1, lanes);
        if (weights.Length != checked(c * m * 9)) throw new ArgumentException("Weight extent");
        int rounded = checked((m + 2 * lanes - 1) / (2 * lanes) * (2 * lanes));
        var result = new float[checked(rounded * c * 9)];
        for (int oc = 0; oc < m; oc++)
            for (int k = 0; k < c * 9; k++)
                result[(oc / lanes * c * 9 + k) * lanes + oc % lanes] = weights[oc * c * 9 + k];
        return result;
    }

    internal static void PackInput(ReadOnlySpan<float> input, Span<float> packed, int c, int h, int w, int lanes)
    {
        int count = checked(c * (h + 2) * (w + 2));
        if (input.Length != checked(c * h * w) || packed.Length < count || input.Overlaps(packed))
            throw new ArgumentException("Input extent/alias");
        packed = packed.Slice(0, count); packed.Clear();
        PackInputTiles(input, packed, c, h, w, lanes);
    }

    internal static bool Execute(ReadOnlySpan<float> input, ReadOnlySpan<float> prepared,
        ReadOnlySpan<float> bias, ReadOnlySpan<float> residual, Span<float> destination,
        Span<float> packedInput, Span<float> packedOutput, int c, int m, int h, int w, int stride, int lanes, bool relu)
    {
        Geometry(c, m, h, w, stride, lanes);
        int oh = Output(h, stride), ow = Output(w, stride), count = checked(m * oh * ow);
        int inputCount = checked(c * (h + 2) * (w + 2));
        int weightCount = checked((m + 2 * lanes - 1) / (2 * lanes) * (2 * lanes) * c * 9);
        if (input.Length != c * h * w || prepared.Length != weightCount || destination.Length != count
            || (bias.Length != 0 && bias.Length != m) || (residual.Length != 0 && residual.Length != count)
            || packedInput.Length < inputCount || packedOutput.Length < count)
            throw new ArgumentException("Buffer extent");
        packedInput = packedInput.Slice(0, inputCount); packedOutput = packedOutput.Slice(0, count);
        if (input.Overlaps(destination) || prepared.Overlaps(destination) || bias.Overlaps(destination) || residual.Overlaps(destination)
            || input.Overlaps(packedInput) || prepared.Overlaps(packedInput) || bias.Overlaps(packedInput) || residual.Overlaps(packedInput)
            || input.Overlaps(packedOutput) || prepared.Overlaps(packedOutput) || bias.Overlaps(packedOutput) || residual.Overlaps(packedOutput)
            || packedInput.Overlaps(packedOutput) || packedInput.Overlaps(destination) || packedOutput.Overlaps(destination))
            throw new ArgumentException("Overlapping buffers");
        if (lanes == 16 && !Avx512F.IsSupported || lanes == 8 && (!Avx2.IsSupported || !Fma.IsSupported))
            throw new PlatformNotSupportedException();
        if (!Finite(input) || !Finite(prepared) || !Finite(bias) || !Finite(residual)) return false;
        PackInput(input, packedInput, c, h, w, lanes);
        // Every output lane must overwrite its sentinel, including row tails.
        fixed (float* x = packedInput, weights = prepared, output = packedOutput)
        {
            if (lanes == 16) Kernel512(x, weights, output, c, m, h, w, stride, oh, ow);
            else Kernel256(x, weights, output, c, m, h, w, stride, oh, ow);
        }
        UnpackEpilogue(packedOutput, destination, bias, residual, m, oh * ow, lanes, relu);
        return true;
    }

    static bool Finite(ReadOnlySpan<float> values)
    {
        var exponent = Vector256.Create(0x7f800000);
        fixed (float* p = values)
        {
            int i = 0;
            for (; i + 8 <= values.Length; i += 8)
                if (Vector256.EqualsAny(((Vector256<float>*)p)[i / 8].AsInt32() & exponent, exponent)) return false;
            for (; i < values.Length; i++) if (!float.IsFinite(p[i])) return false;
        }
        return true;
    }

    static float AddBias(float value, float bias)
    {
        var b = Vector128.CreateScalar(bias);
        return float.IsNaN(bias) ? Sse.AddScalar(b, Vector128<float>.Zero).ToScalar()
            : Sse.AddScalar(Vector128.CreateScalar(value), b).ToScalar();
    }

}
