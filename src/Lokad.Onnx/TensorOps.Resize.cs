namespace Lokad.Onnx;

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

using static Lokad.Onnx.MathOps;
using static Lokad.Onnx.Profiler;

public abstract partial class Tensor<T> : TensorBase, IList, IList<T>, IReadOnlyList<T>, IStructuralComparable, IStructuralEquatable, ITensor, INumericTensor
where T : unmanaged
{
    // Shared Resize preparation: validates the 4D NCHW contract and splits
    // declared and input extents. Parameter names in the throws below are
    // contractual (callers pass rank for input). Scales, coordinates, and
    // rounding stay per-type where precision differs.
    static (int NOut, int COut, int HOut, int WOut, int NIn, int CIn, int HIn, int WIn) ResizeGeometry(int rank, int[] sizes, ReadOnlySpan<int> inputDims)
    {
        if (rank != 4) throw new ArgumentException("input", "Resize currently supports only 4D tensors (NCHW).");
        if (sizes is null || sizes.Length != 4) throw new ArgumentException("sizes", "Resize sizes must be a 1D array of length 4.");
        int nOut = sizes[0];
        int cOut = sizes[1];
        int hOut = sizes[2];
        int wOut = sizes[3];
        int nIn = inputDims[0];
        int cIn = inputDims[1];
        int hIn = inputDims[2];
        int wIn = inputDims[3];
        if (nOut != nIn || cOut != cIn)
        {
            throw new ArgumentException("sizes", "Resize currently requires N and C dimensions to remain unchanged.");
        }
        return (nOut, cOut, hOut, wOut, nIn, cIn, hIn, wIn);
    }

    public static Tensor<float> Resize(Tensor<float> input, int[] sizes, MathOps.ResizeMode mode, MathOps.ResizeCoordinateTransformation coordinateTransformationMode, MathOps.ResizeNearestMode nearestMode, float cubicCoeffA, double[]? scales)
    {
        StartOpStage(OpStage.ValidateArguments);
        var (nOut, cOut, hOut, wOut, nIn, cIn, hIn, wIn) = ResizeGeometry(input.Rank, sizes, input.Dimensions);

        var output = DenseTensor<float>.OfShape(sizes);
        var xd = input.ToDenseTensor();
        var xs = xd.Buffer.Span;
        var os = output.Buffer.Span;
        // Coordinates use the true scales when the caller derived sizes from
        // them: floored sizes would otherwise give back a different scale
        // (verified against ORT 1.29: 5 * 1.5 floors to 7 but samples with 1.5).
        var scaleH = scales is null ? (float)hOut / hIn : (float)scales[2];
        var scaleW = scales is null ? (float)wOut / wIn : (float)scales[3];

        float TransformCoordinate(int outIndex, int inSize, int outSize, float scale)
        {
            return coordinateTransformationMode switch
            {
                MathOps.ResizeCoordinateTransformation.HalfPixel => (outIndex + 0.5f) / scale - 0.5f,
                MathOps.ResizeCoordinateTransformation.AlignCorners => outSize == 1 ? 0f : outIndex * (inSize - 1f) / (outSize - 1f),
                MathOps.ResizeCoordinateTransformation.Asymmetric => outIndex / scale,
                _ => throw new NotSupportedException($"coordinate_transformation_mode {coordinateTransformationMode} is not supported."),
            };
        }

        // round_prefer_floor rounds exact halves down under every coordinate
        // mode: ceil(coord - 0.5) (ORT 1.29 with explicit nearest_mode:
        // asymmetric 0.5 -> 0, align_corners 0.5/1.5 -> 0/1). Integer-scale
        // half_pixel never lands on an exact half, which is why the earlier
        // half-up spelling survived half_pixel tests while asymmetric
        // upscales (which hit halves routinely) diverged.
        int NearestIndex(float coord, int inSize)
        {
            var value = nearestMode switch
            {
                MathOps.ResizeNearestMode.Floor => (int)MathF.Floor(coord),
                MathOps.ResizeNearestMode.Ceil => (int)MathF.Ceiling(coord),
                MathOps.ResizeNearestMode.RoundPreferFloor => (int)MathF.Ceiling(coord - 0.5f),
                MathOps.ResizeNearestMode.RoundPreferCeil => (int)MathF.Floor(coord + 0.5f),
                _ => throw new NotSupportedException($"nearest_mode {nearestMode} is not supported."),
            };
            return Math.Clamp(value, 0, inSize - 1);
        }

        static float CubicWeight(float x, float a)
        {
            var t = MathF.Abs(x);
            if (t <= 1f)
            {
                return ((a + 2f) * t * t * t) - ((a + 3f) * t * t) + 1f;
            }
            if (t < 2f)
            {
                return (a * t * t * t) - (5f * a * t * t) + (8f * a * t) - (4f * a);
            }
            return 0f;
        }

        for (int n = 0; n < nOut; n++)
        {
            for (int c = 0; c < cOut; c++)
            {
                for (int oy = 0; oy < hOut; oy++)
                {
                    var inY = TransformCoordinate(oy, hIn, hOut, scaleH);
                    if (mode == MathOps.ResizeMode.Nearest)
                    {
                        var ny = NearestIndex(inY, hIn);
                        for (int ox = 0; ox < wOut; ox++)
                        {
                            var inX = TransformCoordinate(ox, wIn, wOut, scaleW);
                            var nx = NearestIndex(inX, wIn);
                            os[(((n * cIn) + c) * hOut + oy) * wOut + ox] = xs[(((n * cIn) + c) * hIn + ny) * wIn + nx];
                        }
                    }
                    else if (mode == MathOps.ResizeMode.Linear)
                    {
                        var y0 = MathF.Floor(inY);
                        var y1 = y0 + 1f;
                        var y0i = Math.Clamp((int)y0, 0, hIn - 1);
                        var y1i = Math.Clamp((int)y1, 0, hIn - 1);
                        var ly = inY - y0;
                        var wy0 = 1f - ly;
                        var wy1 = ly;
                        for (int ox = 0; ox < wOut; ox++)
                        {
                            var inX = TransformCoordinate(ox, wIn, wOut, scaleW);
                            var x0 = MathF.Floor(inX);
                            var x1 = x0 + 1f;
                            var x0i = Math.Clamp((int)x0, 0, wIn - 1);
                            var x1i = Math.Clamp((int)x1, 0, wIn - 1);
                            var lx = inX - x0;
                            var wx0 = 1f - lx;
                            var wx1 = lx;
                            int y0Row = (((n * cIn) + c) * hIn + y0i) * wIn;
                            int y1Row = (((n * cIn) + c) * hIn + y1i) * wIn;
                            int dstIdx = (((n * cIn) + c) * hOut + oy) * wOut + ox;
                            // Degenerate taps (same index after clamping) read
                            // their single element with no blend: blending
                            // would multiply it by a zero weight, and 0 * inf
                            // is NaN (ORT 1.29: an exact coordinate hit yields
                            // the hit element). Non-degenerate pixels keep the
                            // exact four-term order.
                            if (x0i == x1i && y0i == y1i) os[dstIdx] = xs[y0Row + x0i];
                            else if (x0i == x1i) os[dstIdx] = xs[y0Row + x0i] * wy0 + xs[y1Row + x0i] * wy1;
                            else if (y0i == y1i) os[dstIdx] = xs[y0Row + x0i] * wx0 + xs[y0Row + x1i] * wx1;
                            else
                            {
                                var v00 = xs[y0Row + x0i];
                                var v01 = xs[y0Row + x1i];
                                var v10 = xs[y1Row + x0i];
                                var v11 = xs[y1Row + x1i];
                                os[dstIdx] = (v00 * wy0 * wx0) + (v01 * wy0 * wx1) + (v10 * wy1 * wx0) + (v11 * wy1 * wx1);
                            }
                        }
                    }
                    else if (mode == MathOps.ResizeMode.Cubic)
                    {
                        var yBase = (int)MathF.Floor(inY);
                        var wy = new float[4];
                        var yIdx = new int[4];
                        for (int i = 0; i < 4; i++)
                        {
                            var yi = yBase - 1 + i;
                            yIdx[i] = Math.Clamp(yi, 0, hIn - 1);
                            wy[i] = CubicWeight(inY - yi, cubicCoeffA);
                        }
                        var wx = new float[4];
                        var xIdx = new int[4];
                        for (int ox = 0; ox < wOut; ox++)
                        {
                            var inX = TransformCoordinate(ox, wIn, wOut, scaleW);
                            var xBase = (int)MathF.Floor(inX);
                            for (int i = 0; i < 4; i++)
                            {
                                var xi = xBase - 1 + i;
                                xIdx[i] = Math.Clamp(xi, 0, wIn - 1);
                                wx[i] = CubicWeight(inX - xi, cubicCoeffA);
                            }
                            var sum = 0f;
                            for (int iy = 0; iy < 4; iy++)
                            {
                                var wyv = wy[iy];
                                for (int ix = 0; ix < 4; ix++)
                                {
                                    sum += wyv * wx[ix] * xs[(((n * cIn) + c) * hIn + yIdx[iy]) * wIn + xIdx[ix]];
                                }
                            }
                            os[(((n * cIn) + c) * hOut + oy) * wOut + ox] = sum;
                        }
                    }
                    else
                    {
                        throw new NotSupportedException($"Resize mode {mode} is not supported.");
                    }
                }
            }
        }

        return output;
    }

    public static Tensor<double> Resize(Tensor<double> input, int[] sizes, MathOps.ResizeMode mode, MathOps.ResizeCoordinateTransformation coordinateTransformationMode, MathOps.ResizeNearestMode nearestMode, double cubicCoeffA, double[]? scales)
    {
        StartOpStage(OpStage.ValidateArguments);
        var (nOut, cOut, hOut, wOut, nIn, cIn, hIn, wIn) = ResizeGeometry(input.Rank, sizes, input.Dimensions);

        var output = DenseTensor<double>.OfShape(sizes);
        var xd = input.ToDenseTensor();
        var xs = xd.Buffer.Span;
        var os = output.Buffer.Span;
        var scaleH = scales is null ? (double)hOut / hIn : scales[2];
        var scaleW = scales is null ? (double)wOut / wIn : scales[3];

        double TransformCoordinate(int outIndex, int inSize, int outSize, double scale)
        {
            return coordinateTransformationMode switch
            {
                MathOps.ResizeCoordinateTransformation.HalfPixel => (outIndex + 0.5) / scale - 0.5,
                MathOps.ResizeCoordinateTransformation.AlignCorners => outSize == 1 ? 0d : outIndex * (inSize - 1d) / (outSize - 1d),
                MathOps.ResizeCoordinateTransformation.Asymmetric => outIndex / scale,
                _ => throw new NotSupportedException($"coordinate_transformation_mode {coordinateTransformationMode} is not supported."),
            };
        }

        // See the float copy: round_prefer_floor rounds halves down in every mode.
        int NearestIndex(double coord, int inSize)
        {
            var value = nearestMode switch
            {
                MathOps.ResizeNearestMode.Floor => (int)Math.Floor(coord),
                MathOps.ResizeNearestMode.Ceil => (int)Math.Ceiling(coord),
                MathOps.ResizeNearestMode.RoundPreferFloor => (int)Math.Ceiling(coord - 0.5),
                MathOps.ResizeNearestMode.RoundPreferCeil => (int)Math.Floor(coord + 0.5),
                _ => throw new NotSupportedException($"nearest_mode {nearestMode} is not supported."),
            };
            return Math.Clamp(value, 0, inSize - 1);
        }

        static double CubicWeight(double x, double a)
        {
            var t = Math.Abs(x);
            if (t <= 1d)
            {
                return ((a + 2d) * t * t * t) - ((a + 3d) * t * t) + 1d;
            }
            if (t < 2d)
            {
                return (a * t * t * t) - (5d * a * t * t) + (8d * a * t) - (4d * a);
            }
            return 0d;
        }

        for (int n = 0; n < nOut; n++)
        {
            for (int c = 0; c < cOut; c++)
            {
                for (int oy = 0; oy < hOut; oy++)
                {
                    var inY = TransformCoordinate(oy, hIn, hOut, scaleH);
                    if (mode == MathOps.ResizeMode.Nearest)
                    {
                        var ny = NearestIndex(inY, hIn);
                        for (int ox = 0; ox < wOut; ox++)
                        {
                            var inX = TransformCoordinate(ox, wIn, wOut, scaleW);
                            var nx = NearestIndex(inX, wIn);
                            os[(((n * cIn) + c) * hOut + oy) * wOut + ox] = xs[(((n * cIn) + c) * hIn + ny) * wIn + nx];
                        }
                    }
                    else if (mode == MathOps.ResizeMode.Linear)
                    {
                        var y0 = Math.Floor(inY);
                        var y1 = y0 + 1d;
                        var y0i = Math.Clamp((int)y0, 0, hIn - 1);
                        var y1i = Math.Clamp((int)y1, 0, hIn - 1);
                        var ly = inY - y0;
                        var wy0 = 1d - ly;
                        var wy1 = ly;
                        for (int ox = 0; ox < wOut; ox++)
                        {
                            var inX = TransformCoordinate(ox, wIn, wOut, scaleW);
                            var x0 = Math.Floor(inX);
                            var x1 = x0 + 1d;
                            var x0i = Math.Clamp((int)x0, 0, wIn - 1);
                            var x1i = Math.Clamp((int)x1, 0, wIn - 1);
                            var lx = inX - x0;
                            var wx0 = 1d - lx;
                            var wx1 = lx;
                            int y0Row = (((n * cIn) + c) * hIn + y0i) * wIn;
                            int y1Row = (((n * cIn) + c) * hIn + y1i) * wIn;
                            int dstIdx = (((n * cIn) + c) * hOut + oy) * wOut + ox;
                            // Degenerate taps (same index after clamping) read
                            // their single element with no blend: blending
                            // would multiply it by a zero weight, and 0 * inf
                            // is NaN (ORT 1.29: an exact coordinate hit yields
                            // the hit element). Non-degenerate pixels keep the
                            // exact four-term order.
                            if (x0i == x1i && y0i == y1i) os[dstIdx] = xs[y0Row + x0i];
                            else if (x0i == x1i) os[dstIdx] = xs[y0Row + x0i] * wy0 + xs[y1Row + x0i] * wy1;
                            else if (y0i == y1i) os[dstIdx] = xs[y0Row + x0i] * wx0 + xs[y0Row + x1i] * wx1;
                            else
                            {
                                var v00 = xs[y0Row + x0i];
                                var v01 = xs[y0Row + x1i];
                                var v10 = xs[y1Row + x0i];
                                var v11 = xs[y1Row + x1i];
                                os[dstIdx] = (v00 * wy0 * wx0) + (v01 * wy0 * wx1) + (v10 * wy1 * wx0) + (v11 * wy1 * wx1);
                            }
                        }
                    }
                    else if (mode == MathOps.ResizeMode.Cubic)
                    {
                        var yBase = (int)Math.Floor(inY);
                        var wy = new double[4];
                        var yIdx = new int[4];
                        for (int i = 0; i < 4; i++)
                        {
                            var yi = yBase - 1 + i;
                            yIdx[i] = Math.Clamp(yi, 0, hIn - 1);
                            wy[i] = CubicWeight(inY - yi, cubicCoeffA);
                        }
                        var wx = new double[4];
                        var xIdx = new int[4];
                        for (int ox = 0; ox < wOut; ox++)
                        {
                            var inX = TransformCoordinate(ox, wIn, wOut, scaleW);
                            var xBase = (int)Math.Floor(inX);
                            for (int i = 0; i < 4; i++)
                            {
                                var xi = xBase - 1 + i;
                                xIdx[i] = Math.Clamp(xi, 0, wIn - 1);
                                wx[i] = CubicWeight(inX - xi, cubicCoeffA);
                            }
                            var sum = 0d;
                            for (int iy = 0; iy < 4; iy++)
                            {
                                var wyv = wy[iy];
                                for (int ix = 0; ix < 4; ix++)
                                {
                                    sum += wyv * wx[ix] * xs[(((n * cIn) + c) * hIn + yIdx[iy]) * wIn + xIdx[ix]];
                                }
                            }
                            os[(((n * cIn) + c) * hOut + oy) * wOut + ox] = sum;
                        }
                    }
                    else
                    {
                        throw new NotSupportedException($"Resize mode {mode} is not supported.");
                    }
                }
            }
        }

        return output;
    }
}
