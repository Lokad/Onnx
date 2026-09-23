using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
namespace Lokad.Onnx;
internal static unsafe partial class ConvBlockedSpatial
{
    static void MultiplyWinograd256(float* v, float* u, float* p, int c, int m)
    {
        const int lanes = 8;
        for (int k = 0; k < 16; k++)
        for (int oc = 0; oc < m; oc += lanes)
        {
            var a0 = Vector256<float>.Zero;
            var a1 = Vector256<float>.Zero;
            var a2 = Vector256<float>.Zero;
            var a3 = Vector256<float>.Zero;
            var a4 = Vector256<float>.Zero;
            var a5 = Vector256<float>.Zero;
            var a6 = Vector256<float>.Zero;
            var a7 = Vector256<float>.Zero;
            float* weights = u + k * c * m + oc;
            float* inputs = v + k * c * WinogradBatch;
            for (int ic = 0; ic < c; ic++)
            {
                var weight = *(Vector256<float>*)weights;
                a0 = Fma.MultiplyAdd(Vector256.Create(inputs[0]), weight, a0);
                a1 = Fma.MultiplyAdd(Vector256.Create(inputs[1]), weight, a1);
                a2 = Fma.MultiplyAdd(Vector256.Create(inputs[2]), weight, a2);
                a3 = Fma.MultiplyAdd(Vector256.Create(inputs[3]), weight, a3);
                a4 = Fma.MultiplyAdd(Vector256.Create(inputs[4]), weight, a4);
                a5 = Fma.MultiplyAdd(Vector256.Create(inputs[5]), weight, a5);
                a6 = Fma.MultiplyAdd(Vector256.Create(inputs[6]), weight, a6);
                a7 = Fma.MultiplyAdd(Vector256.Create(inputs[7]), weight, a7);
                weights += m; inputs += WinogradBatch;
            }
            float* output = p + (k * m + oc) * WinogradBatch;
            *(Vector256<float>*)(output + 0 * lanes) = a0;
            *(Vector256<float>*)(output + 1 * lanes) = a1;
            *(Vector256<float>*)(output + 2 * lanes) = a2;
            *(Vector256<float>*)(output + 3 * lanes) = a3;
            *(Vector256<float>*)(output + 4 * lanes) = a4;
            *(Vector256<float>*)(output + 5 * lanes) = a5;
            *(Vector256<float>*)(output + 6 * lanes) = a6;
            *(Vector256<float>*)(output + 7 * lanes) = a7;
        }
    }
    static void OutputWinograd256(float* p, float* output, int m, int h, int w, int tileWidth, int first, int count)
    {
        const int lanes = 8;
        int spatial = h * w;
        for (int oc = 0; oc < m; oc += lanes)
        for (int tile = 0; tile < count; tile++)
        {
            float* src = p + oc * WinogradBatch + tile * lanes;
            int step = m * WinogradBatch;
            int y = (first + tile) / tileWidth * 2, x = (first + tile) % tileWidth * 2;
            var a0 = Avx.Add(Avx.Add(*(Vector256<float>*)(src + 0 * step), *(Vector256<float>*)(src + 4 * step)), *(Vector256<float>*)(src + 8 * step));
            var b0 = Avx.Subtract(Avx.Subtract(*(Vector256<float>*)(src + 4 * step), *(Vector256<float>*)(src + 8 * step)), *(Vector256<float>*)(src + 12 * step));
            var a1 = Avx.Add(Avx.Add(*(Vector256<float>*)(src + 1 * step), *(Vector256<float>*)(src + 5 * step)), *(Vector256<float>*)(src + 9 * step));
            var b1 = Avx.Subtract(Avx.Subtract(*(Vector256<float>*)(src + 5 * step), *(Vector256<float>*)(src + 9 * step)), *(Vector256<float>*)(src + 13 * step));
            var a2 = Avx.Add(Avx.Add(*(Vector256<float>*)(src + 2 * step), *(Vector256<float>*)(src + 6 * step)), *(Vector256<float>*)(src + 10 * step));
            var b2 = Avx.Subtract(Avx.Subtract(*(Vector256<float>*)(src + 6 * step), *(Vector256<float>*)(src + 10 * step)), *(Vector256<float>*)(src + 14 * step));
            var a3 = Avx.Add(Avx.Add(*(Vector256<float>*)(src + 3 * step), *(Vector256<float>*)(src + 7 * step)), *(Vector256<float>*)(src + 11 * step));
            var b3 = Avx.Subtract(Avx.Subtract(*(Vector256<float>*)(src + 7 * step), *(Vector256<float>*)(src + 11 * step)), *(Vector256<float>*)(src + 15 * step));
            var r00 = Avx.Add(Avx.Add(a0, a1), a2);
            var r01 = Avx.Subtract(Avx.Subtract(a1, a2), a3);
            var r10 = Avx.Add(Avx.Add(b0, b1), b2);
            var r11 = Avx.Subtract(Avx.Subtract(b1, b2), b3);
            float* dst = output + oc * spatial + (y * w + x) * lanes;
            *(Vector256<float>*)dst = r00;
            if (x + 1 < w) *(Vector256<float>*)(dst + lanes) = r01;
            if (y + 1 < h)
            {
                *(Vector256<float>*)(dst + w * lanes) = r10;
                if (x + 1 < w) *(Vector256<float>*)(dst + (w + 1) * lanes) = r11;
            }
        }
    }
    static void MultiplyWinograd512(float* v, float* u, float* p, int c, int m)
    {
        const int lanes = 16;
        for (int k = 0; k < 16; k++)
        for (int oc = 0; oc < m; oc += lanes)
        {
            var a0 = Vector512<float>.Zero;
            var a1 = Vector512<float>.Zero;
            var a2 = Vector512<float>.Zero;
            var a3 = Vector512<float>.Zero;
            var a4 = Vector512<float>.Zero;
            var a5 = Vector512<float>.Zero;
            var a6 = Vector512<float>.Zero;
            var a7 = Vector512<float>.Zero;
            float* weights = u + k * c * m + oc;
            float* inputs = v + k * c * WinogradBatch;
            for (int ic = 0; ic < c; ic++)
            {
                var weight = *(Vector512<float>*)weights;
                a0 = Avx512F.FusedMultiplyAdd(Vector512.Create(inputs[0]), weight, a0);
                a1 = Avx512F.FusedMultiplyAdd(Vector512.Create(inputs[1]), weight, a1);
                a2 = Avx512F.FusedMultiplyAdd(Vector512.Create(inputs[2]), weight, a2);
                a3 = Avx512F.FusedMultiplyAdd(Vector512.Create(inputs[3]), weight, a3);
                a4 = Avx512F.FusedMultiplyAdd(Vector512.Create(inputs[4]), weight, a4);
                a5 = Avx512F.FusedMultiplyAdd(Vector512.Create(inputs[5]), weight, a5);
                a6 = Avx512F.FusedMultiplyAdd(Vector512.Create(inputs[6]), weight, a6);
                a7 = Avx512F.FusedMultiplyAdd(Vector512.Create(inputs[7]), weight, a7);
                weights += m; inputs += WinogradBatch;
            }
            float* output = p + (k * m + oc) * WinogradBatch;
            *(Vector512<float>*)(output + 0 * lanes) = a0;
            *(Vector512<float>*)(output + 1 * lanes) = a1;
            *(Vector512<float>*)(output + 2 * lanes) = a2;
            *(Vector512<float>*)(output + 3 * lanes) = a3;
            *(Vector512<float>*)(output + 4 * lanes) = a4;
            *(Vector512<float>*)(output + 5 * lanes) = a5;
            *(Vector512<float>*)(output + 6 * lanes) = a6;
            *(Vector512<float>*)(output + 7 * lanes) = a7;
        }
    }
    static void OutputWinograd512(float* p, float* output, int m, int h, int w, int tileWidth, int first, int count)
    {
        const int lanes = 16;
        int spatial = h * w;
        for (int oc = 0; oc < m; oc += lanes)
        for (int tile = 0; tile < count; tile++)
        {
            float* src = p + oc * WinogradBatch + tile * lanes;
            int step = m * WinogradBatch;
            int y = (first + tile) / tileWidth * 2, x = (first + tile) % tileWidth * 2;
            var a0 = Avx512F.Add(Avx512F.Add(*(Vector512<float>*)(src + 0 * step), *(Vector512<float>*)(src + 4 * step)), *(Vector512<float>*)(src + 8 * step));
            var b0 = Avx512F.Subtract(Avx512F.Subtract(*(Vector512<float>*)(src + 4 * step), *(Vector512<float>*)(src + 8 * step)), *(Vector512<float>*)(src + 12 * step));
            var a1 = Avx512F.Add(Avx512F.Add(*(Vector512<float>*)(src + 1 * step), *(Vector512<float>*)(src + 5 * step)), *(Vector512<float>*)(src + 9 * step));
            var b1 = Avx512F.Subtract(Avx512F.Subtract(*(Vector512<float>*)(src + 5 * step), *(Vector512<float>*)(src + 9 * step)), *(Vector512<float>*)(src + 13 * step));
            var a2 = Avx512F.Add(Avx512F.Add(*(Vector512<float>*)(src + 2 * step), *(Vector512<float>*)(src + 6 * step)), *(Vector512<float>*)(src + 10 * step));
            var b2 = Avx512F.Subtract(Avx512F.Subtract(*(Vector512<float>*)(src + 6 * step), *(Vector512<float>*)(src + 10 * step)), *(Vector512<float>*)(src + 14 * step));
            var a3 = Avx512F.Add(Avx512F.Add(*(Vector512<float>*)(src + 3 * step), *(Vector512<float>*)(src + 7 * step)), *(Vector512<float>*)(src + 11 * step));
            var b3 = Avx512F.Subtract(Avx512F.Subtract(*(Vector512<float>*)(src + 7 * step), *(Vector512<float>*)(src + 11 * step)), *(Vector512<float>*)(src + 15 * step));
            var r00 = Avx512F.Add(Avx512F.Add(a0, a1), a2);
            var r01 = Avx512F.Subtract(Avx512F.Subtract(a1, a2), a3);
            var r10 = Avx512F.Add(Avx512F.Add(b0, b1), b2);
            var r11 = Avx512F.Subtract(Avx512F.Subtract(b1, b2), b3);
            float* dst = output + oc * spatial + (y * w + x) * lanes;
            *(Vector512<float>*)dst = r00;
            if (x + 1 < w) *(Vector512<float>*)(dst + lanes) = r01;
            if (y + 1 < h)
            {
                *(Vector512<float>*)(dst + w * lanes) = r10;
                if (x + 1 < w) *(Vector512<float>*)(dst + (w + 1) * lanes) = r11;
            }
        }
    }
}
