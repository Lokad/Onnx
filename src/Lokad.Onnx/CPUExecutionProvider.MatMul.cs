namespace Lokad.Onnx;


using System;
using System.Collections.Generic;
using System.Linq;

using static OpResult;
public partial class CPUExecutionProvider
{

    public static OpResult MatMul(ITensor? A, ITensor? B, ExecutionOptions? options, TensorBufferPool? pool)
    {
        var op = OpType.MatMul;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        if (A.ElementType != B.ElementType)
        {
            return WrongInputType(op, nameof(B), "Input tensors must be of the same type.", B);
        }

        var opts = (options ?? ExecutionOptions.Default).Validated();
        switch (A.ElementType)
        {
            case TensorElementType.Int32: return Success(op, Tensor<int>.MatMul(SpeedDensify((Tensor<int>)A, opts, OpStage.CopyX), SpeedDensify((Tensor<int>)B, opts, OpStage.CopyY), opts.Tensor));
            case TensorElementType.Float: return Success(op, Tensor<float>.MatMul(SpeedDensify((Tensor<float>)A, opts, OpStage.CopyX), SpeedDensify((Tensor<float>)B, opts, OpStage.CopyY), opts.Tensor, pool));
            case TensorElementType.Double: return Success(op, Tensor<double>.MatMul(SpeedDensify((Tensor<double>)A, opts, OpStage.CopyX), SpeedDensify((Tensor<double>)B, opts, OpStage.CopyY), opts.Tensor));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    /// <summary>
    /// Scale-fused MatMul: computes MatMul(Mul(A, scale), B) with the scale
    /// applied live per execution. Composite sequencing first: the exact
    /// legacy Mul then MatMul run in order, so fusion plumbing lands with
    /// identical values before any fused kernel arrives. Inputs are
    /// [data, weights, scalar-scale]; any rank the legacy paths accept works.
    /// </summary>
    public static OpResult ScaledMatMul(ITensor? A, ITensor? B, ITensor? Scale, ExecutionOptions? options, TensorBufferPool? pool)
    {
        var op = OpType.ScaledMatMul;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        if (Scale is null) return MissingInput(op, nameof(Scale));
        var mul = Mul(A, Scale, options, pool);
        if (mul.Status != OpStatus.Success || mul.Outputs is null || mul.Outputs.Length != 1 || mul.Outputs[0] is null)
            return mul;
        var mm = MatMul(mul.Outputs[0], B, options, pool);
        if (mm.Status != OpStatus.Success || mm.Outputs is null)
            return mm;
        return Success(op, mm.Outputs);
    }

    /// <summary>
    /// Trailing-scale MatMul: computes Div(MatMul(A, B), divisor) with the
    /// exact legacy sequence, so the fusion lands with identical values
    /// before any alpha-epilogue kernel arrives (E5-1 M2). Internal: the
    /// optimizer is the only producer of trailing ScaledMatMul nodes.
    /// </summary>
    internal static OpResult ScaledMatMulTrailing(ITensor? A, ITensor? B, ITensor? Divisor, ExecutionOptions? options, TensorBufferPool? pool)
    {
        var op = OpType.ScaledMatMul;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        if (Divisor is null) return MissingInput(op, nameof(Divisor));
        var opts = (options ?? ExecutionOptions.Default).Validated();
        if (TryRunTrailingAlpha(A, B, Divisor, opts, out var alphaOut) && alphaOut is not null)
            return Success(op, alphaOut);
        var mm = MatMul(A, B, options, pool);
        if (mm.Status != OpStatus.Success || mm.Outputs is null || mm.Outputs.Length != 1 || mm.Outputs[0] is null)
            return mm;
        var dv = Div(mm.Outputs[0], Divisor, options, pool);
        if (dv.Status != OpStatus.Success || dv.Outputs is null)
            return dv;
        return Success(op, dv.Outputs);
    }

    /// <summary>
    /// Epilogue-alpha fast path for trailing Div fusion (E5-1 M2): when the
    /// divisor is a readable finite nonzero float, the shapes route to the
    /// unpacked tiled kernel, and alpha is finite, the twin scales each
    /// element once on its single store. Divisor 1 skips the Div entirely
    /// (x/1 == x bitwise). Every other shape falls back to the identical
    /// composite, preserving exact IEEE behavior there unconditionally.
    /// </summary>
    static bool TryRunTrailingAlpha(ITensor? A, ITensor? B, ITensor? Divisor, ExecutionOptions opts, out ITensor? output)
    {
        output = null;
        var topts = opts.Tensor;
        if (!topts.UseSimd || !topts.UseIntrinsics || !System.Runtime.Intrinsics.X86.Fma.IsSupported) return false;
        if (A is not Tensor<float> fa || B is not Tensor<float> fb) return false;
        if (Divisor is not DenseTensor<float> dd || dd.Length != 1) return false;
        float d = dd.Buffer.Span[0];
        if (fa.Rank != 2 || fb.Rank != 2) return false;
        int m = fa.Dimensions[0], n = fa.Dimensions[1], k = fb.Dimensions[1];
        if (fb.Dimensions[0] != n) return false;
        if (d == 1f)
        {
            var mm1 = MatMul(A, B, opts, null);
            if (mm1.Status != OpStatus.Success || mm1.Outputs is null || mm1.Outputs.Length != 1 || mm1.Outputs[0] is null)
                return false;
            output = mm1.Outputs[0];
            return true;
        }
        if (!float.IsFinite(d) || d == 0f) return false;
        float alpha = 1f / d;
        if (!float.IsFinite(alpha)) return false;
        // Tiled-kernel territory only (mirrors RunFloatMatMulKernel limits);
        // packed, classic, scalar and m1 shapes keep the composite.
        if ((m % 2) != 0 || m < 2 || n >= 2560 || k >= 2560) return false;
        var xa = Tensor<float>.RequireContiguous(fa, nameof(A), topts.CopyReporter);
        var xb = Tensor<float>.RequireContiguous(fb, nameof(B), topts.CopyReporter);
        var dest = DenseTensor<float>.OfShape(m, k);
        unsafe
        {
            using var xh = xa.Buffer.Pin();
            using var yh = xb.Buffer.Pin();
            using var oh = dest.Buffer.Pin();
            MathOps.mm_unsafe_vectorized_intrinsics_2x4tiled_alpha(m, n, k, (float*)xh.Pointer, (float*)yh.Pointer, (float*)oh.Pointer, alpha);
        }
        output = dest;
        return true;
    }

    static Tensor<T> SpeedDensify<T>(Tensor<T> t, ExecutionOptions opts, OpStage stage) where T : unmanaged
    {
        // Speed mode densifies operands up front (views materialize here,
        // counted); other modes leave views for the kernels, which densify
        // defensively and count through the same choke point.
        if (opts.Optimization != OptimizationMode.Speed) return t;
        // A view over a fresh packed clone must reach the packed kernel
        // intact: materializing it would freeze scrambled bytes row-major.
        if (typeof(T) == typeof(float) && t is Tensor<float> tf
            && opts.Tensor.PackedMatMulWeights is not null
            && GraphPacking.ResolvePacked(opts.Tensor.PackedMatMulWeights, tf) is not null)
            return t;
        Profiler.StartOpStage(stage);
        return Tensor<T>.RequireContiguous(t, nameof(t), opts.Tensor.CopyReporter);
    }

    public static OpResult Gemm(ITensor? A, ITensor? B, ITensor? C, float alpha, float beta, ExecutionOptions? options, int transA, int transB)
    {
        var op = OpType.Gemm;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        if (A.ElementType != B.ElementType)
        {
            return WrongInputType(op, nameof(B), "Input tensors must be of the same type.", B);
        }
        var opts = (options ?? ExecutionOptions.Default).Validated();
        if (transA != 0 && transA != 1) return AttributeNotSupported(op, "transA", transA.ToString(), "transA must be 0 or 1.");
        if (transB != 0 && transB != 1) return AttributeNotSupported(op, "transB", transB.ToString(), "transB must be 0 or 1.");
        if (A.Rank != 2) return WrongInputShape(op, nameof(A), 2, A);
        if (B.Rank != 2) return WrongInputShape(op, nameof(B), 2, B);
        if (C is not null && C.ElementType != A.ElementType) return WrongInputType(op, nameof(C), A.ElementType, C, "Gemm input C must have the same element type as A and B.");
        int m = transA == 1 ? A.Dims[1] : A.Dims[0];
        int kA = transA == 1 ? A.Dims[0] : A.Dims[1];
        int kB = transB == 1 ? B.Dims[1] : B.Dims[0];
        int n = transB == 1 ? B.Dims[0] : B.Dims[1];
        if (kA != kB) return WrongInputShape(op, nameof(B), B, "Gemm inner dimensions disagree after transpose.");
        int k = kA;
        if (C is not null && C.Length > 1)
        {
            bool okc = (C.Rank == 1 && C.Dims[0] == n)
                || (C.Rank == 2 && ((C.Dims[0] == 1 && C.Dims[1] == n) || (C.Dims[0] == m && C.Dims[1] == 1) || (C.Dims[0] == m && C.Dims[1] == n)));
            if (!okc) return WrongInputShape(op, nameof(C), C, "Gemm bias C must be a scalar, [N], [1,N], [M,1], or [M,N].");
        }

        /// <summary>
        /// Writes alpha * (a @ b) + beta * c into one owned destination: the product
        /// lands directly in the result through the shared overwriting entry, and the
        /// scale/bias pass runs only when it can change anything.
        /// </summary>
        static Tensor<float> GemmFloat(Tensor<float> a, Tensor<float> b, Tensor<float>? c, int m, int k, int n, float alpha, float beta, TensorExecutionOptions tensorOptions)
        {
            var y = DenseTensor<float>.OfShape(m, n);
            Tensor<float>.MatMul2D(a, b, y, tensorOptions);
            if (alpha == 1f && (c is null || beta == 0f)) return y;
            var ys = y.Buffer.Span;
            if (c is null || beta == 0f)
            {
                for (int i = 0; i < ys.Length; i++) ys[i] = alpha * ys[i] + beta * 0f;
                return y;
            }
            var cbias = c.ToArray();
            if (c.Length == 1)
            {
                float cb = cbias[0];
                for (int i = 0; i < ys.Length; i++) ys[i] = alpha * ys[i] + beta * cb;
                return y;
            }
            if (c.Rank == 1)
            {
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++) ys[i * n + j] = alpha * ys[i * n + j] + beta * cbias[j];
                }
                return y;
            }
            int crows = c.Dimensions[0];
            int ccols = c.Dimensions[1];
            for (int i = 0; i < m; i++)
            {
                int r = (crows == 1 ? 0 : i) * ccols;
                for (int j = 0; j < n; j++) ys[i * n + j] = alpha * ys[i * n + j] + beta * cbias[r + (ccols == 1 ? 0 : j)];
            }
            return y;
        }

        /// <summary>
        /// Writes alpha * (a @ b) + beta * c into one owned destination: the product
        /// lands directly in the result through the shared overwriting entry, and the
        /// scale/bias pass runs only when it can change anything.
        /// </summary>
        static Tensor<double> GemmDouble(Tensor<double> a, Tensor<double> b, Tensor<double>? c, int m, int k, int n, float alpha, float beta, TensorExecutionOptions tensorOptions)
        {
            var y = DenseTensor<double>.OfShape(m, n);
            Tensor<double>.MatMul2D(a, b, y, tensorOptions);
            if (alpha == 1f && (c is null || beta == 0f)) return y;
            var ys = y.Buffer.Span;
            if (c is null || beta == 0f)
            {
                for (int i = 0; i < ys.Length; i++) ys[i] = alpha * ys[i] + beta * 0.0;
                return y;
            }
            var cbias = c.ToArray();
            if (c.Length == 1)
            {
                double cb = cbias[0];
                for (int i = 0; i < ys.Length; i++) ys[i] = alpha * ys[i] + beta * cb;
                return y;
            }
            if (c.Rank == 1)
            {
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++) ys[i * n + j] = alpha * ys[i * n + j] + beta * cbias[j];
                }
                return y;
            }
            int crows = c.Dimensions[0];
            int ccols = c.Dimensions[1];
            for (int i = 0; i < m; i++)
            {
                int r = (crows == 1 ? 0 : i) * ccols;
                for (int j = 0; j < n; j++) ys[i * n + j] = alpha * ys[i * n + j] + beta * cbias[r + (ccols == 1 ? 0 : j)];
            }
            return y;
        }
        switch (A.ElementType)
        {
            case TensorElementType.Float:
            {
                var ea = transA == 1 ? Tensor<float>.Transpose((Tensor<float>)A, null) : (Tensor<float>)A;
                var eb = transB == 1 ? Tensor<float>.Transpose((Tensor<float>)B, null) : (Tensor<float>)B;
                return Success(op, GemmFloat(ea, eb, (Tensor<float>?)C, m, k, n, alpha, beta, opts.Tensor));
            }
            case TensorElementType.Double:
            {
                var ea = transA == 1 ? Tensor<double>.Transpose((Tensor<double>)A, null) : (Tensor<double>)A;
                var eb = transB == 1 ? Tensor<double>.Transpose((Tensor<double>)B, null) : (Tensor<double>)B;
                return Success(op, GemmDouble(ea, eb, (Tensor<double>?)C, m, k, n, alpha, beta, opts.Tensor));
            }
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    /// <summary>
    /// Gemm with GELU epilogue: runs the identical Gemm computation, then applies
    /// GeluSpanFloat over the owned exact-fit float output in place, which is bitwise
    /// identical to separate Gemm and Gelu nodes by construction. Honors the opt-in
    /// tanh trial switch with the vector tanh epilogue. Non-float or unexpected
    /// layouts take the legacy two-step with identical values.
    /// </summary>
    public static OpResult GemmGelu(ITensor? A, ITensor? B, ITensor? C, float alpha, float beta, ExecutionOptions? options, int transA, int transB, string? approximate)
    {
        var op = OpType.GemmGelu;
        var r = Gemm(A, B, C, alpha, beta, options, transA, transB);
        if (r.Status != OpStatus.Success || r.Outputs is null || r.Outputs.Length != 1)
            return r;
        if (r.Outputs[0] is DenseTensor<float> df && df.Buffer.Length == (int)df.Length)
        {
            if (approximate == "tanh" || UseGeluTanhTrial())
                Tensor<float>.GeluTanhSpanFloat(df.Buffer.Span, df.Buffer.Span);
            else
                Tensor<float>.GeluSpanFloat(df.Buffer.Span, df.Buffer.Span);
            r.Op = op;
            return r;
        }
        var g = Gelu(r.Outputs[0], null, options, null);
        if (g.Status != OpStatus.Success || g.Outputs is null || g.Outputs.Length != 1)
            return g;
        var o = g.Outputs[0];
        r.Outputs = new ITensor[] { o };
        r.Op = op;
        return r;
    }
}
