using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx;

internal static unsafe partial class ConvBlockedSpatial
{
    // Execute has already verified finite operands, extents and all aliases.
    // The 8x8 shuffle sequence matches MathOps' existing pure transpose kernel.
    static void UnpackEpilogue(ReadOnlySpan<float> blocked, Span<float> destination,
        ReadOnlySpan<float> bias, ReadOnlySpan<float> residual, int m, int spatial, int lanes, bool relu)
    {
        fixed (float* input = blocked, output = destination, b = bias, r = residual)
        {
            for (int block = 0; block < m; block += lanes)
                for (int half = 0; half < lanes; half += 8)
                {
                    float* source = input + block * spatial + half;
                    int channel = block + half, p = 0;
                    for (; p + 8 <= spatial; p += 8)
                    {
                        float* tile = source + p * lanes;
                        var r0 = *(Vector256<float>*)(tile + 0 * lanes);
                        var r1 = *(Vector256<float>*)(tile + 1 * lanes);
                        var r2 = *(Vector256<float>*)(tile + 2 * lanes);
                        var r3 = *(Vector256<float>*)(tile + 3 * lanes);
                        var r4 = *(Vector256<float>*)(tile + 4 * lanes);
                        var r5 = *(Vector256<float>*)(tile + 5 * lanes);
                        var r6 = *(Vector256<float>*)(tile + 6 * lanes);
                        var r7 = *(Vector256<float>*)(tile + 7 * lanes);
                        var t0 = Avx.UnpackLow(r0, r1); var t1 = Avx.UnpackHigh(r0, r1);
                        var t2 = Avx.UnpackLow(r2, r3); var t3 = Avx.UnpackHigh(r2, r3);
                        var t4 = Avx.UnpackLow(r4, r5); var t5 = Avx.UnpackHigh(r4, r5);
                        var t6 = Avx.UnpackLow(r6, r7); var t7 = Avx.UnpackHigh(r6, r7);
                        var e0 = Avx.Shuffle(t0, t2, 0x44); var e1 = Avx.Shuffle(t0, t2, 0xEE);
                        var e2 = Avx.Shuffle(t1, t3, 0x44); var e3 = Avx.Shuffle(t1, t3, 0xEE);
                        var e4 = Avx.Shuffle(t4, t6, 0x44); var e5 = Avx.Shuffle(t4, t6, 0xEE);
                        var e6 = Avx.Shuffle(t5, t7, 0x44); var e7 = Avx.Shuffle(t5, t7, 0xEE);
                        StoreEpilogue8(Avx.Permute2x128(e0, e4, 0x20), output, b, r, channel + 0, spatial, p, relu);
                        StoreEpilogue8(Avx.Permute2x128(e1, e5, 0x20), output, b, r, channel + 1, spatial, p, relu);
                        StoreEpilogue8(Avx.Permute2x128(e2, e6, 0x20), output, b, r, channel + 2, spatial, p, relu);
                        StoreEpilogue8(Avx.Permute2x128(e3, e7, 0x20), output, b, r, channel + 3, spatial, p, relu);
                        StoreEpilogue8(Avx.Permute2x128(e0, e4, 0x31), output, b, r, channel + 4, spatial, p, relu);
                        StoreEpilogue8(Avx.Permute2x128(e1, e5, 0x31), output, b, r, channel + 5, spatial, p, relu);
                        StoreEpilogue8(Avx.Permute2x128(e2, e6, 0x31), output, b, r, channel + 6, spatial, p, relu);
                        StoreEpilogue8(Avx.Permute2x128(e3, e7, 0x31), output, b, r, channel + 7, spatial, p, relu);
                    }
                    for (int lane = 0; lane < 8; lane++)
                        for (int tail = p; tail < spatial; tail++)
                        {
                            float value = source[tail * lanes + lane];
                            if (b != null) value = AddBias(value, b[channel + lane]);
                            int index = (channel + lane) * spatial + tail;
                            if (r != null) value += r[index];
                            if (relu) value = value <= 0f ? (value == 0f ? value : 0f) : value;
                            output[index] = value;
                        }
                }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void StoreEpilogue8(Vector256<float> value, float* output, float* bias, float* residual,
        int channel, int spatial, int p, bool relu)
    {
        int index = channel * spatial + p;
        if (bias != null) value = Avx.Add(value, Vector256.Create(bias[channel]));
        if (residual != null) value = Avx.Add(value, *(Vector256<float>*)(residual + index));
        if (relu)
        {
            var negative = Avx.Compare(value, Vector256<float>.Zero, FloatComparisonMode.OrderedLessThanNonSignaling);
            value = Avx.AndNot(negative, value); // Clear strictly negative lanes; retain -0 and NaNs.
        }
        *(Vector256<float>*)(output + index) = value;
    }
}
