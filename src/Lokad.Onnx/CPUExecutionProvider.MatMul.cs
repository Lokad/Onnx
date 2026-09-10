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
            case TensorElementType.Int32: return Success(op, Tensor<int>.MatMul(SpeedDensify((Tensor<int>)A, opts), SpeedDensify((Tensor<int>)B, opts), opts.Tensor));
            case TensorElementType.Float: return Success(op, Tensor<float>.MatMul(SpeedDensify((Tensor<float>)A, opts), SpeedDensify((Tensor<float>)B, opts), opts.Tensor, pool));
            case TensorElementType.Double: return Success(op, Tensor<double>.MatMul(SpeedDensify((Tensor<double>)A, opts), SpeedDensify((Tensor<double>)B, opts), opts.Tensor));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    static Tensor<T> SpeedDensify<T>(Tensor<T> t, ExecutionOptions opts) where T : unmanaged
    {
        // Speed mode densifies operands up front (views materialize here,
        // counted); other modes leave views for the kernels, which densify
        // defensively and count through the same choke point.
        if (opts.Optimization != OptimizationMode.Speed) return t;
        Profiler.StartOpStage(OpStage.Copy);
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
}
