using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
namespace Lokad.Onnx;

internal static unsafe partial class ConvBlockedSpatial
{
    static void Kernel256(float* x, float* weights, float* output, int c, int m, int h, int w, int stride, int oh, int ow)
    {
        const int lanes = 8;
        int ph = h + 2, pw = w + 2, spatial = oh * ow, fusedEnd = spatial / 8 * 8;
        for (int oc = 0; oc < m; oc += 2 * lanes)
        for (int y = 0; y < oh; y++)
        {
            int col = 0;
            for (; col + 6 <= ow && y * ow + col + 6 <= fusedEnd; col += 6)
            {
                Vector256<float> a00 = Vector256<float>.Zero;
                Vector256<float> a01 = Vector256<float>.Zero;
                Vector256<float> a10 = Vector256<float>.Zero;
                Vector256<float> a11 = Vector256<float>.Zero;
                Vector256<float> a20 = Vector256<float>.Zero;
                Vector256<float> a21 = Vector256<float>.Zero;
                Vector256<float> a30 = Vector256<float>.Zero;
                Vector256<float> a31 = Vector256<float>.Zero;
                Vector256<float> a40 = Vector256<float>.Zero;
                Vector256<float> a41 = Vector256<float>.Zero;
                Vector256<float> a50 = Vector256<float>.Zero;
                Vector256<float> a51 = Vector256<float>.Zero;
                float* w0 = weights + oc * c * 9;
                float* w1 = w0 + lanes * c * 9;
                for (int ic = 0; ic < c; ic++)
                for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++)
                {
                    float* input = x + ((ic / lanes * ph + y * stride + ky) * pw + col * stride + kx) * lanes + ic % lanes;
                    var wv0 = *(Vector256<float>*)w0; var wv1 = *(Vector256<float>*)w1;
                    var i0 = Vector256.Create(input[0 * stride * lanes]);
                    a00 = Fma.MultiplyAdd(i0, wv0, a00);
                    a01 = Fma.MultiplyAdd(i0, wv1, a01);
                    var i1 = Vector256.Create(input[1 * stride * lanes]);
                    a10 = Fma.MultiplyAdd(i1, wv0, a10);
                    a11 = Fma.MultiplyAdd(i1, wv1, a11);
                    var i2 = Vector256.Create(input[2 * stride * lanes]);
                    a20 = Fma.MultiplyAdd(i2, wv0, a20);
                    a21 = Fma.MultiplyAdd(i2, wv1, a21);
                    var i3 = Vector256.Create(input[3 * stride * lanes]);
                    a30 = Fma.MultiplyAdd(i3, wv0, a30);
                    a31 = Fma.MultiplyAdd(i3, wv1, a31);
                    var i4 = Vector256.Create(input[4 * stride * lanes]);
                    a40 = Fma.MultiplyAdd(i4, wv0, a40);
                    a41 = Fma.MultiplyAdd(i4, wv1, a41);
                    var i5 = Vector256.Create(input[5 * stride * lanes]);
                    a50 = Fma.MultiplyAdd(i5, wv0, a50);
                    a51 = Fma.MultiplyAdd(i5, wv1, a51);
                    w0 += lanes; w1 += lanes;
                }
                *(Vector256<float>*)(output + (oc / lanes * spatial + y * ow + col + 0) * lanes) = a00;
                if (oc + lanes < m) *(Vector256<float>*)(output + ((oc / lanes + 1) * spatial + y * ow + col + 0) * lanes) = a01;
                *(Vector256<float>*)(output + (oc / lanes * spatial + y * ow + col + 1) * lanes) = a10;
                if (oc + lanes < m) *(Vector256<float>*)(output + ((oc / lanes + 1) * spatial + y * ow + col + 1) * lanes) = a11;
                *(Vector256<float>*)(output + (oc / lanes * spatial + y * ow + col + 2) * lanes) = a20;
                if (oc + lanes < m) *(Vector256<float>*)(output + ((oc / lanes + 1) * spatial + y * ow + col + 2) * lanes) = a21;
                *(Vector256<float>*)(output + (oc / lanes * spatial + y * ow + col + 3) * lanes) = a30;
                if (oc + lanes < m) *(Vector256<float>*)(output + ((oc / lanes + 1) * spatial + y * ow + col + 3) * lanes) = a31;
                *(Vector256<float>*)(output + (oc / lanes * spatial + y * ow + col + 4) * lanes) = a40;
                if (oc + lanes < m) *(Vector256<float>*)(output + ((oc / lanes + 1) * spatial + y * ow + col + 4) * lanes) = a41;
                *(Vector256<float>*)(output + (oc / lanes * spatial + y * ow + col + 5) * lanes) = a50;
                if (oc + lanes < m) *(Vector256<float>*)(output + ((oc / lanes + 1) * spatial + y * ow + col + 5) * lanes) = a51;
            }
            for (; col < ow; col++)
            {
                var a0 = Vector256<float>.Zero; var a1 = Vector256<float>.Zero;
                float* w0 = weights + oc * c * 9; float* w1 = w0 + lanes * c * 9;
                bool fused = y * ow + col < fusedEnd;
                for (int ic = 0; ic < c; ic++)
                for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++)
                {
                    float value = x[((ic / lanes * ph + y * stride + ky) * pw + col * stride + kx) * lanes + ic % lanes];
                    var input = Vector256.Create(value);
                    var wv0 = *(Vector256<float>*)w0; var wv1 = *(Vector256<float>*)w1;
                    if (fused)
                    {
                        a0 = Fma.MultiplyAdd(input, wv0, a0); a1 = Fma.MultiplyAdd(input, wv1, a1);
                    }
                    else
                    {
                        a0 = a0 + input * wv0; a1 = a1 + input * wv1;
                    }
                    w0 += lanes; w1 += lanes;
                }
                *(Vector256<float>*)(output + (oc / lanes * spatial + y * ow + col) * lanes) = a0;
                if (oc + lanes < m) *(Vector256<float>*)(output + ((oc / lanes + 1) * spatial + y * ow + col) * lanes) = a1;
            }
        }
    }
    static void Kernel512(float* x, float* weights, float* output, int c, int m, int h, int w, int stride, int oh, int ow)
    {
        const int lanes = 16;
        int ph = h + 2, pw = w + 2, spatial = oh * ow, fusedEnd = spatial / 8 * 8;
        for (int oc = 0; oc < m; oc += 2 * lanes)
        for (int y = 0; y < oh; y++)
        {
            int col = 0;
            for (; col + 6 <= ow && y * ow + col + 6 <= fusedEnd; col += 6)
            {
                Vector512<float> a00 = Vector512<float>.Zero;
                Vector512<float> a01 = Vector512<float>.Zero;
                Vector512<float> a10 = Vector512<float>.Zero;
                Vector512<float> a11 = Vector512<float>.Zero;
                Vector512<float> a20 = Vector512<float>.Zero;
                Vector512<float> a21 = Vector512<float>.Zero;
                Vector512<float> a30 = Vector512<float>.Zero;
                Vector512<float> a31 = Vector512<float>.Zero;
                Vector512<float> a40 = Vector512<float>.Zero;
                Vector512<float> a41 = Vector512<float>.Zero;
                Vector512<float> a50 = Vector512<float>.Zero;
                Vector512<float> a51 = Vector512<float>.Zero;
                float* w0 = weights + oc * c * 9;
                float* w1 = w0 + lanes * c * 9;
                for (int ic = 0; ic < c; ic++)
                for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++)
                {
                    float* input = x + ((ic / lanes * ph + y * stride + ky) * pw + col * stride + kx) * lanes + ic % lanes;
                    var wv0 = *(Vector512<float>*)w0; var wv1 = *(Vector512<float>*)w1;
                    var i0 = Vector512.Create(input[0 * stride * lanes]);
                    a00 = Avx512F.FusedMultiplyAdd(i0, wv0, a00);
                    a01 = Avx512F.FusedMultiplyAdd(i0, wv1, a01);
                    var i1 = Vector512.Create(input[1 * stride * lanes]);
                    a10 = Avx512F.FusedMultiplyAdd(i1, wv0, a10);
                    a11 = Avx512F.FusedMultiplyAdd(i1, wv1, a11);
                    var i2 = Vector512.Create(input[2 * stride * lanes]);
                    a20 = Avx512F.FusedMultiplyAdd(i2, wv0, a20);
                    a21 = Avx512F.FusedMultiplyAdd(i2, wv1, a21);
                    var i3 = Vector512.Create(input[3 * stride * lanes]);
                    a30 = Avx512F.FusedMultiplyAdd(i3, wv0, a30);
                    a31 = Avx512F.FusedMultiplyAdd(i3, wv1, a31);
                    var i4 = Vector512.Create(input[4 * stride * lanes]);
                    a40 = Avx512F.FusedMultiplyAdd(i4, wv0, a40);
                    a41 = Avx512F.FusedMultiplyAdd(i4, wv1, a41);
                    var i5 = Vector512.Create(input[5 * stride * lanes]);
                    a50 = Avx512F.FusedMultiplyAdd(i5, wv0, a50);
                    a51 = Avx512F.FusedMultiplyAdd(i5, wv1, a51);
                    w0 += lanes; w1 += lanes;
                }
                *(Vector512<float>*)(output + (oc / lanes * spatial + y * ow + col + 0) * lanes) = a00;
                if (oc + lanes < m) *(Vector512<float>*)(output + ((oc / lanes + 1) * spatial + y * ow + col + 0) * lanes) = a01;
                *(Vector512<float>*)(output + (oc / lanes * spatial + y * ow + col + 1) * lanes) = a10;
                if (oc + lanes < m) *(Vector512<float>*)(output + ((oc / lanes + 1) * spatial + y * ow + col + 1) * lanes) = a11;
                *(Vector512<float>*)(output + (oc / lanes * spatial + y * ow + col + 2) * lanes) = a20;
                if (oc + lanes < m) *(Vector512<float>*)(output + ((oc / lanes + 1) * spatial + y * ow + col + 2) * lanes) = a21;
                *(Vector512<float>*)(output + (oc / lanes * spatial + y * ow + col + 3) * lanes) = a30;
                if (oc + lanes < m) *(Vector512<float>*)(output + ((oc / lanes + 1) * spatial + y * ow + col + 3) * lanes) = a31;
                *(Vector512<float>*)(output + (oc / lanes * spatial + y * ow + col + 4) * lanes) = a40;
                if (oc + lanes < m) *(Vector512<float>*)(output + ((oc / lanes + 1) * spatial + y * ow + col + 4) * lanes) = a41;
                *(Vector512<float>*)(output + (oc / lanes * spatial + y * ow + col + 5) * lanes) = a50;
                if (oc + lanes < m) *(Vector512<float>*)(output + ((oc / lanes + 1) * spatial + y * ow + col + 5) * lanes) = a51;
            }
            for (; col < ow; col++)
            {
                var a0 = Vector512<float>.Zero; var a1 = Vector512<float>.Zero;
                float* w0 = weights + oc * c * 9; float* w1 = w0 + lanes * c * 9;
                bool fused = y * ow + col < fusedEnd;
                for (int ic = 0; ic < c; ic++)
                for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++)
                {
                    float value = x[((ic / lanes * ph + y * stride + ky) * pw + col * stride + kx) * lanes + ic % lanes];
                    var input = Vector512.Create(value);
                    var wv0 = *(Vector512<float>*)w0; var wv1 = *(Vector512<float>*)w1;
                    if (fused)
                    {
                        a0 = Avx512F.FusedMultiplyAdd(input, wv0, a0); a1 = Avx512F.FusedMultiplyAdd(input, wv1, a1);
                    }
                    else
                    {
                        a0 = a0 + input * wv0; a1 = a1 + input * wv1;
                    }
                    w0 += lanes; w1 += lanes;
                }
                *(Vector512<float>*)(output + (oc / lanes * spatial + y * ow + col) * lanes) = a0;
                if (oc + lanes < m) *(Vector512<float>*)(output + ((oc / lanes + 1) * spatial + y * ow + col) * lanes) = a1;
            }
        }
    }
}
