namespace Lokad.Onnx;

using System.Numerics;
using System.Runtime.Intrinsics;

/// <summary>
/// Mixed-radix forward transform for lengths composed of factors two, three and five.
/// Each SIMD lane carries an independent time frame; roots are immutable and scratch
/// belongs to the caller, so prepared transforms can be shared by graph contexts.
/// </summary>
internal sealed class FrameFourierTransform
{
    internal int Length { get; }
    readonly Vector<float>[] realRoots, imaginaryRoots;
    internal long RetainedBytes => 2L * Length * Vector<float>.Count * 4;

    internal FrameFourierTransform(int length)
    {
        if (length < 1 || length > 4096) throw new ArgumentOutOfRangeException(nameof(length));
        int remaining = length;
        foreach (int factor in new[] { 2, 3, 5 }) while (remaining % factor == 0) remaining /= factor;
        if (remaining != 1) throw new ArgumentException("Unsupported smooth length");
        Length = length; realRoots = new Vector<float>[length]; imaginaryRoots = new Vector<float>[length];
        for (int i = 0; i < length; i++)
        {
            double angle = -2 * Math.PI * i / length;
            realRoots[i] = new((float)Math.Cos(angle)); imaginaryRoots[i] = new((float)Math.Sin(angle));
        }
    }

    internal void Forward(Vector<float>[] inputR, Vector<float>[] inputI, Vector<float>[] outputR, Vector<float>[] outputI)
    {
        if (inputR.Length < Length || inputI.Length < Length || outputR.Length < Length || outputI.Length < Length
            || ReferenceEquals(inputR, outputR) || ReferenceEquals(inputR, outputI)
            || ReferenceEquals(inputI, outputR) || ReferenceEquals(inputI, outputI)
            || ReferenceEquals(outputR, outputI)) throw new ArgumentException("Independent complete arrays required");
        Transform(inputR, inputI, 0, 1, outputR, outputI, 0, Length, 1);
    }

    void Transform(Vector<float>[] inputR, Vector<float>[] inputI, int origin, int stride,
        Vector<float>[] outputR, Vector<float>[] outputI, int start, int n, int rootStride)
    {
        if (n == 1) { outputR[start] = inputR[origin]; outputI[start] = inputI[origin]; return; }
        int radix = n % 2 == 0 ? 2 : n % 3 == 0 ? 3 : 5, m = n / radix;
        if (m != 1) for (int j = 0; j < radix; j++) Transform(inputR, inputI, origin + j * stride, stride * radix, outputR, outputI, start + j * m, m, rootStride * radix);
        for (int k = 0; k < m; k++)
        {
            var ar = m == 1 ? inputR[origin] : outputR[start + k]; var ai = m == 1 ? inputI[origin] : outputI[start + k];
            var br = m == 1 ? inputR[origin + stride] : outputR[start + m + k]; var bi = m == 1 ? inputI[origin + stride] : outputI[start + m + k];
            if (k != 0)
            {
                var r = realRoots[k * rootStride]; var i = imaginaryRoots[k * rootStride];
                var temporary = br * r - bi * i; bi = br * i + bi * r; br = temporary;
            }
            if (radix == 2)
            {
                outputR[start + k] = ar + br; outputI[start + k] = ai + bi;
                outputR[start + m + k] = ar - br; outputI[start + m + k] = ai - bi; continue;
            }
            var cr = m == 1 ? inputR[origin + 2 * stride] : outputR[start + 2 * m + k]; var ci = m == 1 ? inputI[origin + 2 * stride] : outputI[start + 2 * m + k];
            if (k != 0)
            {
                var r = realRoots[2 * k * rootStride]; var i = imaginaryRoots[2 * k * rootStride];
                var temporary = cr * r - ci * i; ci = cr * i + ci * r; cr = temporary;
            }
            if (radix == 3)
            {
                var sr = br + cr; var si = bi + ci; var centerR = ar - new Vector<float>(.5f) * sr; var centerI = ai - new Vector<float>(.5f) * si;
                var rotateR = (bi - ci) * new Vector<float>(.86602540378443864676f);
                var rotateI = (cr - br) * new Vector<float>(.86602540378443864676f);
                outputR[start + k] = ar + sr; outputI[start + k] = ai + si;
                outputR[start + m + k] = centerR + rotateR; outputI[start + m + k] = centerI + rotateI;
                outputR[start + 2 * m + k] = centerR - rotateR; outputI[start + 2 * m + k] = centerI - rotateI; continue;
            }
            var dr = m == 1 ? inputR[origin + 3 * stride] : outputR[start + 3 * m + k]; var di = m == 1 ? inputI[origin + 3 * stride] : outputI[start + 3 * m + k];
            var er = m == 1 ? inputR[origin + 4 * stride] : outputR[start + 4 * m + k]; var ei = m == 1 ? inputI[origin + 4 * stride] : outputI[start + 4 * m + k];
            if (k != 0)
            {
                var r = realRoots[3 * k * rootStride]; var i = imaginaryRoots[3 * k * rootStride];
                var temporary = dr * r - di * i; di = dr * i + di * r; dr = temporary;
                r = realRoots[4 * k * rootStride]; i = imaginaryRoots[4 * k * rootStride];
                temporary = er * r - ei * i; ei = er * i + ei * r; er = temporary;
            }
            var s14r = br + er; var s14i = bi + ei; var s23r = cr + dr; var s23i = ci + di;
            var d14r = br - er; var d14i = bi - ei; var d23r = cr - dr; var d23i = ci - di;
            var center1r = ar + new Vector<float>(.30901699437494742410f) * s14r - new Vector<float>(.80901699437494742410f) * s23r;
            var center1i = ai + new Vector<float>(.30901699437494742410f) * s14i - new Vector<float>(.80901699437494742410f) * s23i;
            var center2r = ar - new Vector<float>(.80901699437494742410f) * s14r + new Vector<float>(.30901699437494742410f) * s23r;
            var center2i = ai - new Vector<float>(.80901699437494742410f) * s14i + new Vector<float>(.30901699437494742410f) * s23i;
            var odd1r = new Vector<float>(.95105651629515357212f) * d14r + new Vector<float>(.58778525229247312917f) * d23r;
            var odd1i = new Vector<float>(.95105651629515357212f) * d14i + new Vector<float>(.58778525229247312917f) * d23i;
            var odd2r = new Vector<float>(.58778525229247312917f) * d14r - new Vector<float>(.95105651629515357212f) * d23r;
            var odd2i = new Vector<float>(.58778525229247312917f) * d14i - new Vector<float>(.95105651629515357212f) * d23i;
            outputR[start + k] = ar + s14r + s23r; outputI[start + k] = ai + s14i + s23i;
            outputR[start + m + k] = center1r + odd1i; outputI[start + m + k] = center1i - odd1r;
            outputR[start + 4 * m + k] = center1r - odd1i; outputI[start + 4 * m + k] = center1i + odd1r;
            outputR[start + 2 * m + k] = center2r + odd2i; outputI[start + 2 * m + k] = center2i - odd2r;
            outputR[start + 3 * m + k] = center2r - odd2i; outputI[start + 3 * m + k] = center2i + odd2r;
        }
    }

    internal void Forward(Vector512<float>[] inputR, Vector512<float>[] inputI, Vector512<float>[] outputR, Vector512<float>[] outputI)
    {
        if (inputR.Length < Length || inputI.Length < Length || outputR.Length < Length || outputI.Length < Length
            || ReferenceEquals(inputR, outputR) || ReferenceEquals(inputR, outputI)
            || ReferenceEquals(inputI, outputR) || ReferenceEquals(inputI, outputI)
            || ReferenceEquals(outputR, outputI)) throw new ArgumentException("Independent complete arrays required");
        Transform(inputR, inputI, 0, 1, outputR, outputI, 0, Length, 1);
    }

    void Transform(Vector512<float>[] inputR, Vector512<float>[] inputI, int origin, int stride,
        Vector512<float>[] outputR, Vector512<float>[] outputI, int start, int n, int rootStride)
    {
        if (n == 1) { outputR[start] = inputR[origin]; outputI[start] = inputI[origin]; return; }
        int radix = n % 2 == 0 ? 2 : n % 3 == 0 ? 3 : 5, m = n / radix;
        if (m != 1) for (int j = 0; j < radix; j++) Transform(inputR, inputI, origin + j * stride, stride * radix, outputR, outputI, start + j * m, m, rootStride * radix);
        for (int k = 0; k < m; k++)
        {
            var ar = m == 1 ? inputR[origin] : outputR[start + k]; var ai = m == 1 ? inputI[origin] : outputI[start + k];
            var br = m == 1 ? inputR[origin + stride] : outputR[start + m + k]; var bi = m == 1 ? inputI[origin + stride] : outputI[start + m + k];
            if (k != 0)
            {
                var r = Vector512.Create(realRoots[k * rootStride][0]); var i = Vector512.Create(imaginaryRoots[k * rootStride][0]);
                var temporary = br * r - bi * i; bi = br * i + bi * r; br = temporary;
            }
            if (radix == 2)
            {
                outputR[start + k] = ar + br; outputI[start + k] = ai + bi;
                outputR[start + m + k] = ar - br; outputI[start + m + k] = ai - bi; continue;
            }
            var cr = m == 1 ? inputR[origin + 2 * stride] : outputR[start + 2 * m + k]; var ci = m == 1 ? inputI[origin + 2 * stride] : outputI[start + 2 * m + k];
            if (k != 0)
            {
                var r = Vector512.Create(realRoots[2 * k * rootStride][0]); var i = Vector512.Create(imaginaryRoots[2 * k * rootStride][0]);
                var temporary = cr * r - ci * i; ci = cr * i + ci * r; cr = temporary;
            }
            if (radix == 3)
            {
                var sr = br + cr; var si = bi + ci; var centerR = ar - Vector512.Create(.5f) * sr; var centerI = ai - Vector512.Create(.5f) * si;
                var rotateR = (bi - ci) * Vector512.Create(.86602540378443864676f);
                var rotateI = (cr - br) * Vector512.Create(.86602540378443864676f);
                outputR[start + k] = ar + sr; outputI[start + k] = ai + si;
                outputR[start + m + k] = centerR + rotateR; outputI[start + m + k] = centerI + rotateI;
                outputR[start + 2 * m + k] = centerR - rotateR; outputI[start + 2 * m + k] = centerI - rotateI; continue;
            }
            var dr = m == 1 ? inputR[origin + 3 * stride] : outputR[start + 3 * m + k]; var di = m == 1 ? inputI[origin + 3 * stride] : outputI[start + 3 * m + k];
            var er = m == 1 ? inputR[origin + 4 * stride] : outputR[start + 4 * m + k]; var ei = m == 1 ? inputI[origin + 4 * stride] : outputI[start + 4 * m + k];
            if (k != 0)
            {
                var r = Vector512.Create(realRoots[3 * k * rootStride][0]); var i = Vector512.Create(imaginaryRoots[3 * k * rootStride][0]);
                var temporary = dr * r - di * i; di = dr * i + di * r; dr = temporary;
                r = Vector512.Create(realRoots[4 * k * rootStride][0]); i = Vector512.Create(imaginaryRoots[4 * k * rootStride][0]);
                temporary = er * r - ei * i; ei = er * i + ei * r; er = temporary;
            }
            var s14r = br + er; var s14i = bi + ei; var s23r = cr + dr; var s23i = ci + di;
            var d14r = br - er; var d14i = bi - ei; var d23r = cr - dr; var d23i = ci - di;
            var center1r = ar + Vector512.Create(.30901699437494742410f) * s14r - Vector512.Create(.80901699437494742410f) * s23r;
            var center1i = ai + Vector512.Create(.30901699437494742410f) * s14i - Vector512.Create(.80901699437494742410f) * s23i;
            var center2r = ar - Vector512.Create(.80901699437494742410f) * s14r + Vector512.Create(.30901699437494742410f) * s23r;
            var center2i = ai - Vector512.Create(.80901699437494742410f) * s14i + Vector512.Create(.30901699437494742410f) * s23i;
            var odd1r = Vector512.Create(.95105651629515357212f) * d14r + Vector512.Create(.58778525229247312917f) * d23r;
            var odd1i = Vector512.Create(.95105651629515357212f) * d14i + Vector512.Create(.58778525229247312917f) * d23i;
            var odd2r = Vector512.Create(.58778525229247312917f) * d14r - Vector512.Create(.95105651629515357212f) * d23r;
            var odd2i = Vector512.Create(.58778525229247312917f) * d14i - Vector512.Create(.95105651629515357212f) * d23i;
            outputR[start + k] = ar + s14r + s23r; outputI[start + k] = ai + s14i + s23i;
            outputR[start + m + k] = center1r + odd1i; outputI[start + m + k] = center1i - odd1r;
            outputR[start + 4 * m + k] = center1r - odd1i; outputI[start + 4 * m + k] = center1i + odd1r;
            outputR[start + 2 * m + k] = center2r + odd2i; outputI[start + 2 * m + k] = center2i - odd2r;
            outputR[start + 3 * m + k] = center2r - odd2i; outputI[start + 3 * m + k] = center2i + odd2r;
        }
    }
}
