namespace Lokad.Onnx;

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

using static Lokad.Onnx.MathOps;
using static Lokad.Onnx.Profiler;

public abstract partial class Tensor<T> : TensorBase, IList, IList<T>, IReadOnlyList<T>, IStructuralComparable, IStructuralEquatable, ITensor, INumericTensor
where T : unmanaged
{
    // Shared N-D MatMul preparation: validates ranks, builds the broadcast
    // plan, and promotes vector operands. Batched execution stays per-dtype.
    static (MatMulShapes.Plan plan, Tensor<TElement> px, Tensor<TElement> py) PlanMatMul<TElement>(Tensor<TElement> x, Tensor<TElement> y) where TElement : unmanaged
    {
        if (x.Rank == 0 || y.Rank == 0) throw new ArgumentException("The rank of each tensor in matrix multiplication must be at least 1; rank-1 vectors are promoted.");
        var plan = MatMulShapes.Create(x.Dimensions, y.Dimensions);
        var px = plan.PromoteX ? x.InsertDim(0) : x;
        var py = plan.PromoteY ? y.InsertDim(y.Rank) : y;
        return (plan, px, py);
    }

    // Shared N-D batched MatMul preparation for the inline paths: validates
    // inner dims, plans the broadcast, materializes the destination,
    // densifies operands, and derives batch geometry. Broadcast order follows
    // the double path (the int path reported the same message either way).
    // Pinning, profiler stages, and kernel loops stay with the callers; the
    // float path already factors this through its batched core.
    static (Tensor<TElement> bx, Tensor<TElement> by, DenseTensor<TElement> z, int[] batchDims, int[] xSteps, int[] ySteps, int[] zSteps, int batchCount, int m, int n, int k) PlanBatchedMatMul<TElement>(
        Tensor<TElement> px, Tensor<TElement> py, ICopyAccountant? copy) where TElement : unmanaged
    {
        var xdl = px.Dimensions[^2..];
        var ydl = py.Dimensions[^2..];
        if (xdl[1] != ydl[0])
        {
            throw new ArgumentException($"The number of columns in the first matrix ({xdl[1]}) is not equal to the number of rows in the second matrix ({ydl[0]}).");
        }
        if (!BroadcastShape(px.Dimensions[0..^2], py.Dimensions[0..^2], out var bd))
        {
            throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
        }
        var bdx = bd.Append(xdl[0]).Append(xdl[1]).ToArray();
        if (!Tensor<TElement>.Broadcast(px, bdx, out var bx))
        {
            throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
        }
        var bdy = bd.Append(ydl[0]).Append(ydl[1]).ToArray();
        if (!Tensor<TElement>.Broadcast(py, bdy, out var by))
        {
            throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
        }
        var z = DenseTensor<TElement>.OfShape(bd.Append(xdl[0]).Append(ydl[1]).ToArray());
        var cbx = RequireContiguous(bx, nameof(bx), copy);
        var cby = RequireContiguous(by, nameof(by), copy);
        bx = cbx;
        by = cby;
        var batchDims = bx.Dimensions[0..^2].ToArray();
        var xSteps = BatchSteps(batchDims, bx.Dimensions, bx.strides);
        var ySteps = BatchSteps(batchDims, by.Dimensions, by.strides);
        var zSteps = BatchSteps(batchDims, z.Dimensions, z.strides);
        int batchCount = BatchCount(batchDims);
        return (bx, by, z, batchDims, xSteps, ySteps, zSteps, batchCount, bx.Dimensions[^2], bx.Dimensions[^1], by.Dimensions[^1]);
    }

    public static Tensor<int> MatMul2D(Tensor<int> x, Tensor<int> y) => MatMul2D(x, y, TensorExecutionOptions.Auto);

    public static Tensor<int> MatMul2D(Tensor<int> x, Tensor<int> y, TensorExecutionOptions options)
    {
        options.Validate();
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException("The number of columns in the first matrix is not equal to the number of rows in the second matrix.");
        var m = x.Dimensions[0];
        var n = x.Dimensions[1];
        var k = y.Dimensions[1];

        var dx = x as DenseTensor<int>;
        var dy = y as DenseTensor<int>;
        var _x = dx is not null && !dx.IsReversedStride ? dx : CountedCopy(x.ToDenseTensor(), options.CopyReporter);
        var _y = dy is not null && !dy.IsReversedStride ? dy : CountedCopy(y.ToDenseTensor(), options.CopyReporter);
        var output = DenseTensor<int>.OfShape(new int[] { x.Dimensions[0], y.Dimensions[1] });

        var xh = _x.Buffer.Pin();
        var yh = _y.Buffer.Pin();
        var oh = output.Buffer.Pin();
        if (options.UseSimd)
        {
            unsafe
            {
                mm_unsafe_vectorized(m, n, k, (int*)xh.Pointer, (int*)yh.Pointer, (int*)oh.Pointer);
            }
        }
        else
        {
            unsafe
            {
                mm(m, n, k, (int*)xh.Pointer, (int*)yh.Pointer, (int*)oh.Pointer);
            }
        }

        xh.Dispose();
        yh.Dispose();
        oh.Dispose();
        return output;
    }

    static (DenseTensor<float> x, DenseTensor<float> y) DensifyFloatOperands(Tensor<float> x, Tensor<float> y, ICopyAccountant? copy)
    {
        var dx = x as DenseTensor<float>;
        var dy = y as DenseTensor<float>;
        if (dx is not { IsReversedStride: false } || dy is not { IsReversedStride: false })
        {
            StartOpStage(OpStage.Copy);
        }
        DenseTensor<float> ddx = dx is { IsReversedStride: false } ownX ? ownX : CountedCopy(x.ToDenseTensor(), copy);
        DenseTensor<float> ddy = dy is { IsReversedStride: false } ownY ? ownY : CountedCopy(y.ToDenseTensor(), copy);
        return (ddx, ddy);
    }

    /// <summary>
    /// Selects the panel-packed kernel for a B-side operand resolving to a
    /// fresh packed clone. Every unmet condition falls back to the unpacked
    /// path, so Scalar and Simd modes, missing FMA, odd row counts outside exact
    /// 3-row groups, and stale mappings keep reading original row-major bytes
    /// by construction. Three-row groups (P65) cover odd multiples of 3 exactly.
    /// </summary>
    static DenseTensor<float>? ResolvePackedKernel(TensorExecutionOptions options, Tensor<float> y, int m)
    {
        if (!options.UseSimd || !options.UseIntrinsics || !Fma.IsSupported) return null;
        // Row groups cover every count of two or more rows; single rows stay
        // unpacked. Non-AVX512 hardware still admits only even and exact
        // 3-row counts, whose decomposition reproduces the legacy calls.
        if (m < 2) return null;
        if (!Avx512F.IsSupported && (m & 1) != 0 && (m % 3) != 0) return null;
        var found = GraphPacking.ResolvePacked(options.PackedMatMulWeights, y);
        if (found is null || found.Dimensions.Length != 2) return null;
        int n = found.Dimensions[0], k = found.Dimensions[1];
        if (!GraphPacking.IsPackableShape(n, k)) return null;
        return found;
    }

    // Tier0-stuck leaf (see PLAN qdA2): force Tier1; few benchmark calls never trip promotion.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static unsafe void RunFloatMatMulKernel(int m, int n, int k, float* x, float* y, float* output, TensorExecutionOptions options, bool overwriteDestination)
    {
        // Register-tiled accumulation wins while both the reduction axis (n)
        // and the output width (k) fit the fast caches: a 25-shape old-vs-new
        // probe wins every shape with n and k below 2560 and loses every shape
        // with n or k at or above 3072, so wider shapes keep the proven kernel.
        const int TiledMatMulAxisLimit = 2560;
        // Panel packing pays off once the M/2 row-group re-reads amortize the
        // single pack pass: a shape probe loses below M=48 and wins from M=64
        // (-22 to -38 percent with pack cost included), so smaller blocks keep
        // the unpacked tiled kernel bit-identically. At 64 rows and above the
        // packed kernel also beats the old kernel at any width (the old output
        // traffic dominates once panels fix the strided reads), so wide axes
        // route packed too; the pool-size guard keeps rental sane.
        const int TiledPackMinRows = 64;
        const long TiledPackMaxElements = 67108864;
        // Overwrite contract: panel-packed kernels fully overwrite their
        // destination, so an uninitialized output skips the clear; every
        // other lane accumulates and keeps the explicit clear. The odd-m
        // non-AVX512 transient leaves its last row to the unpacked fixup,
        // so it stays on the clearing path. This sits above the dispatch
        // so single-row and non-SIMD lanes clear too.
        bool widePackable = (long)n * k <= TiledPackMaxElements;
        bool p65Full = (m % 3) == 0 && m >= TiledPackMinRows && widePackable;
        bool blockPackFull = !p65Full && (m - (m % 2)) >= TiledPackMinRows && widePackable
            && (Avx512F.IsSupported || (m & 1) == 0);
        bool packedFull = options.UseSimd && options.UseIntrinsics && Fma.IsSupported && m >= 2
            && (p65Full || blockPackFull);
        if (overwriteDestination && !packedFull)
            new Span<float>(output, m * k).Clear();
        if (options.UseSimd && options.UseIntrinsics && Fma.IsSupported && m >= 2)
        {
            int blocked = m - (m % 2);
            // On AVX512 hardware the transient pack routes through the same
            // 12-row composer as persistent weights: every piece shares panel
            // layout and per-element FMA order, so the swap stays bitwise
            // identical while sharing each B vector up to twelve ways.
            bool transientAvxCovered = false;
            if ((m % 3) == 0 && m >= TiledPackMinRows && (long)n * k <= TiledPackMaxElements)
            {
                // P65: exact 3-row groups cover every row, so no scalar fixup follows.
                float[] packed = RentScratch<float>(n * k, options);
                try
                {
                    fixed (float* pp = packed)
                    {
                        PackPanelsB(n, k, y, pp);
                        if (Avx512F.IsSupported)
                        {
                            RunPackedRowGroups(m, n, k, x, pp, output, overwrite: true);
                            transientAvxCovered = true;
                        }
                        else
                        {
                            mm_unsafe_vectorized_intrinsics_3x4packed(m, n, k, x, pp, output, overwrite: true);
                        }
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(packed);
                }
            }
            else if (blocked >= TiledPackMinRows && (long)n * k <= TiledPackMaxElements)
            {
                float[] packed = RentScratch<float>(n * k, options);
                try
                {
                    fixed (float* pp = packed)
                    {
                        PackPanelsB(n, k, y, pp);
                        if (Avx512F.IsSupported)
                        {
                            RunPackedRowGroups(m, n, k, x, pp, output, overwrite: true);
                            transientAvxCovered = true;
                        }
                        else
                        {
                            mm_unsafe_vectorized_intrinsics_2x4packed(blocked, n, k, x, pp, output, overwrite: true);
                        }
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(packed);
                }
            }
            else if (n < TiledMatMulAxisLimit && k < TiledMatMulAxisLimit)
            {
                mm_unsafe_vectorized_intrinsics_2x4tiled(blocked, n, k, x, y, output);
            }
            else
            {
                mm_unsafe_vectorized_intrinsics_2x4(blocked, n, k, x, y, output);
            }
            // The P65 3-row branch above already covers every row exactly, so the 2-row
            // remainder fixup must not re-accumulate the last row.
            bool threeRowCovered = (m % 3) == 0 && m >= TiledPackMinRows && (long)n * k <= TiledPackMaxElements;
            if (blocked != m && !threeRowCovered && !transientAvxCovered)
            {
                mm_unsafe_vectorized_intrinsics(1, n, k, x + blocked * n, y, output + blocked * k);
            }
        }
        else if (options.UseSimd && options.UseIntrinsics && Fma.IsSupported)
        {
            mm_unsafe_vectorized_intrinsics(m, n, k, x, y, output);
        }
        else if (options.UseSimd)
        {
            mm_unsafe_vectorized(m, n, k, x, y, output);
        }
        else
        {
            mm(m, n, k, x, y, output);
        }
    }

    public static Tensor<float> MatMul2D(Tensor<float> x, Tensor<float> y) => MatMul2D(x, y, TensorExecutionOptions.Auto);

    public static Tensor<float> MatMul2D(Tensor<float> x, Tensor<float> y, TensorExecutionOptions options)
    {
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        return MatMul2DCore(x, y, DenseTensor<float>.OfShape(new int[] { x.Dimensions[0], y.Dimensions[1] }), options, overwriteDestination: false);
    }

    /// <summary>
    /// Writes the 2D float matrix product into an existing dense destination,
    /// overwriting it. The destination must not alias either input. The
    /// panel-packed kernels overwrite it directly; unpacked fallback lanes
    /// clear it first. Callers may pass uninitialized storage.
    /// </summary>
    public static Tensor<float> MatMul2D(Tensor<float> x, Tensor<float> y, DenseTensor<float> destination, TensorExecutionOptions options)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        return MatMul2DCore(x, y, destination, options, overwriteDestination: true);
    }

    // Tier0-stuck leaf (see PLAN qdA2): force Tier1; few benchmark calls never trip promotion.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    /// <summary>
    /// Runs the panel-packed product with an already-resolved packed B clone
    /// (same kernels and overwrite contract as the MatMul2D packed branch).
    /// The packed tensor carries panel dims [n,k]; x is [m,n], destination
    /// [m,k]. Callers resolve freshness themselves and fall back to the
    /// unpacked path when no prepared clone applies.
    /// </summary>
    internal static Tensor<float> MatMulPackedProduct(Tensor<float> x, DenseTensor<float> packedB, DenseTensor<float> destination, TensorExecutionOptions options)
    {
        options.Validate();
        if (packedB.Rank != 2) throw new ArgumentException(nameof(packedB), "Packed B must be rank 2 with panel dims [n,k].");
        var m = x.Dimensions[0];
        var n = packedB.Dimensions[0];
        var k = packedB.Dimensions[1];
        if (x.Dimensions[1] != n) throw new ArgumentException("X columns must match packed panel rows.");
        if (destination.Dimensions.Length != 2 || destination.Dimensions[0] != m || destination.Dimensions[1] != k) throw new ArgumentException(nameof(destination), "Destination shape must match the matrix product shape.");
        if (!HasStandardStrides(destination)) throw new ArgumentException(nameof(destination), "Destination must have standard row-major strides.");
        if (ReferenceEquals(destination, x) || TensorAlias.SharesBackingMemory(destination, x)) throw new ArgumentException(nameof(destination), "Destination must not alias the input matrix.");
        var dx = RequireContiguous(x, nameof(x), options.CopyReporter);
        StartOpStage(OpStage.Math);
        using var xh = dx.Buffer.Pin();
        using var ph = packedB.Buffer.Pin();
        using var oh = destination.Buffer.Pin();
        unsafe
        {
            // Row groups with 12/6-row AVX512 heads and 3/2-row tails (P5);
            // exact shapes keep single calls bit-identically.
            RunPackedRowGroups(m, n, k, (float*)xh.Pointer, (float*)ph.Pointer, (float*)oh.Pointer, overwrite: true);
        }
        return destination;
    }

    static Tensor<float> MatMul2DCore(Tensor<float> x, Tensor<float> y, DenseTensor<float> destination, TensorExecutionOptions options, bool overwriteDestination)
    {
        options.Validate();
        StartOpStage(OpStage.ValidateArguments);
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException($"The number of columns in the first matrix ({x.Dimensions[1]}) is not equal to the number of rows in the second matrix ({y.Dimensions[0]}).");
        if (destination.Dimensions.Length != 2 || destination.Dimensions[0] != x.Dimensions[0] || destination.Dimensions[1] != y.Dimensions[1]) throw new ArgumentException(nameof(destination), "Destination shape must match the matrix product shape.");
        if (!HasStandardStrides(destination)) throw new ArgumentException(nameof(destination), "Destination must have standard row-major strides.");
        if (ReferenceEquals(destination, x) || ReferenceEquals(destination, y) || TensorAlias.SharesBackingMemory(destination, x) || TensorAlias.SharesBackingMemory(destination, y)) throw new ArgumentException(nameof(destination), "Destination must not alias the input matrices.");
        var m = x.Dimensions[0];
        var n = x.Dimensions[1];
        var k = y.Dimensions[1];

        if (ResolvePackedKernel(options, y, m) is { } packedB)
        {
            return MatMulPackedProduct(x, packedB, destination, options);
        }

        var (_x, _y) = DensifyFloatOperands(x, y, options.CopyReporter);

        StartOpStage(OpStage.Math);
        int rowDop = options.MaxDegreeOfParallelism < 2 || m < 64
            ? 1
            : Math.Min(options.MaxDegreeOfParallelism, m);
        if (rowDop > 1)
        {
            int chunk = (m + rowDop - 1) / rowDop;
            Parallel.For(0, rowDop, new ParallelOptions { MaxDegreeOfParallelism = rowDop }, w =>
            {
                int start = w * chunk;
                int rows = Math.Min(chunk, m - start);
                if (rows <= 0) return;
                using var xh = _x.Buffer.Pin();
                using var yh = _y.Buffer.Pin();
                using var oh = destination.Buffer.Pin();
                unsafe
                {
                    RunFloatMatMulKernel(rows, n, k,
                        (float*)xh.Pointer + start * n,
                        (float*)yh.Pointer,
                        (float*)oh.Pointer + start * k, options, overwriteDestination);
                }
            });
        }
        else
        {
            using var xh = _x.Buffer.Pin();
            using var yh = _y.Buffer.Pin();
            using var oh = destination.Buffer.Pin();
            unsafe
            {
                RunFloatMatMulKernel(m, n, k, (float*)xh.Pointer, (float*)yh.Pointer, (float*)oh.Pointer, options, overwriteDestination);
            }
        }
        return destination;
    }

    /// <summary>Computes the 2D float matrix product, renting the output from the pool when provided.</summary>
    public static Tensor<float> MatMul2D(Tensor<float> x, Tensor<float> y, TensorExecutionOptions options, TensorBufferPool? pool)
    {
        if (pool is null) return MatMul2D(x, y, options);
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        var dims = new int[] { x.Dimensions[0], y.Dimensions[1] };
        int flat;
        checked { flat = dims[0] * dims[1]; }
        var destination = new DenseTensor<float>(new Memory<float>(pool.Rent<float>(flat)), dims);
        return MatMul2DCore(x, y, destination, options, overwriteDestination: true);
    }

    public static Tensor<double> MatMul2D(Tensor<double> x, Tensor<double> y) => MatMul2D(x, y, TensorExecutionOptions.Auto);

    public static Tensor<double> MatMul2D(Tensor<double> x, Tensor<double> y, TensorExecutionOptions options)
    {
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        return MatMul2DCoreDouble(x, y, DenseTensor<double>.OfShape(new int[] { x.Dimensions[0], y.Dimensions[1] }), options, clearDestination: false);
    }

    /// <summary>
    /// Writes the 2D double matrix product into an existing dense destination,
    /// overwriting it. The destination must not alias either input. The raw
    /// kernels accumulate, so this entry point clears the destination first.
    /// </summary>
    public static Tensor<double> MatMul2D(Tensor<double> x, Tensor<double> y, DenseTensor<double> destination, TensorExecutionOptions options)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        return MatMul2DCoreDouble(x, y, destination, options, clearDestination: true);
    }

    static Tensor<double> MatMul2DCoreDouble(Tensor<double> x, Tensor<double> y, DenseTensor<double> destination, TensorExecutionOptions options, bool clearDestination)
    {
        options.Validate();
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException("The number of columns in the first matrix is not equal to the number of rows in the second matrix.");
        if (destination.Dimensions.Length != 2 || destination.Dimensions[0] != x.Dimensions[0] || destination.Dimensions[1] != y.Dimensions[1]) throw new ArgumentException(nameof(destination), "Destination shape must match the matrix product shape.");
        if (!HasStandardStrides(destination)) throw new ArgumentException(nameof(destination), "Destination must have standard row-major strides.");
        if (ReferenceEquals(destination, x) || ReferenceEquals(destination, y) || TensorAlias.SharesBackingMemory(destination, x) || TensorAlias.SharesBackingMemory(destination, y)) throw new ArgumentException(nameof(destination), "Destination must not alias the input matrices.");
        if (clearDestination) destination.Buffer.Span.Clear();
        var m = x.Dimensions[0];
        var n = x.Dimensions[1];
        var k = y.Dimensions[1];


        var dx = x as DenseTensor<double>;
        var dy = y as DenseTensor<double>;
        var _x = dx is not null && !dx.IsReversedStride ? dx : CountedCopy(x.ToDenseTensor(), options.CopyReporter);
        var _y = dy is not null && !dy.IsReversedStride ? dy : CountedCopy(y.ToDenseTensor(), options.CopyReporter);
        var output = destination;

        var xh = _x.Buffer.Pin();
        var yh = _y.Buffer.Pin();
        var oh = output.Buffer.Pin();
        if (options.UseSimd && options.UseIntrinsics && Fma.IsSupported)
        {
            unsafe
            {
                mm_unsafe_vectorized_intrinsics(m, n, k, (double*)xh.Pointer, (double*)yh.Pointer, (double*)oh.Pointer);
            }
        }
        else if (options.UseSimd)
        {
            unsafe
            {
                mm_unsafe_vectorized(m, n, k, (double*)xh.Pointer, (double*)yh.Pointer, (double*)oh.Pointer);
            }
        }
        else
            unsafe
            {
                mm(m, n, k, (double*)xh.Pointer, (double*)yh.Pointer, (double*)oh.Pointer);
            }
        xh.Dispose();
        yh.Dispose();
        oh.Dispose();
        return output;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static Tensor<int> MatMul(Tensor<int> x, Tensor<int> y) => MatMul(x, y, TensorExecutionOptions.Auto);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static Tensor<int> MatMul(Tensor<int> x, Tensor<int> y, TensorExecutionOptions options)
    
    {
        var (plan, px, py) = PlanMatMul(x, y);
        Tensor<int> core;
        if (px.Rank == 2 && py.Rank == 2)
        {
            core = Tensor<int>.MatMul2D(px, py, options);
        }
        else
        {
            var (bx, by, z, batchDims, xSteps, ySteps, zSteps, batchCount, m, n, k) = PlanBatchedMatMul(px, py, options.CopyReporter);
            using var xh = bx.Storage.Pin();
            using var yh = by.Storage.Pin();
            using var zh = z.Storage.Pin();
            StartOpStage(OpStage.Math);
            unsafe
            {
                var xp = (int*)xh.Pointer;
                var yp = (int*)yh.Pointer;
                var zp = (int*)zh.Pointer;
                int r = batchDims.Length;
                var coords = new int[r];
                int ox = 0, oy = 0, oz = 0;
                for (int b = 0; b < batchCount; b++)
                {
                    if (options.UseSimd)
                    {
                        mm_unsafe_vectorized(m, n, k, xp + ox, yp + oy, zp + oz);
                    }
                    else
                    {
                        mm(m, n, k, xp + ox, yp + oy, zp + oz);
                    }
                    for (int d = r - 1; d >= 0; d--)
                    {
                        coords[d]++;
                        ox += xSteps[d]; oy += ySteps[d]; oz += zSteps[d];
                        if (coords[d] < batchDims[d]) break;
                        coords[d] = 0;
                        ox -= xSteps[d] * batchDims[d]; oy -= ySteps[d] * batchDims[d]; oz -= zSteps[d] * batchDims[d];
                    }
                }
            }
            core = z;
        }
        return MatMulShapes.Squeeze(core, plan);
    }


    static int[] MatMulOutputShape(ReadOnlySpan<int> xd, ReadOnlySpan<int> yd) => MatMulShapes.Create(xd, yd).OutputShape;

    /// <summary>
    /// Flat batch strides for one standard-dense operand: zero where the operand
    /// reuses a batch entry, flat stride otherwise. Matches GetStorageIndex on
    /// the same coordinates for densified operands.
    /// </summary>
    static int[] BatchSteps(ReadOnlySpan<int> batchDims, ReadOnlySpan<int> operandDims, int[] operandStrides)
    {
        var steps = new int[batchDims.Length];
        for (int d = 0; d < batchDims.Length; d++)
            steps[d] = (d < operandDims.Length - 2 && operandDims[d] != 1) ? operandStrides[d] : 0;
        return steps;
    }

    static int BatchCount(ReadOnlySpan<int> batchDims)
    {
        int n = 1;
        foreach (var d in batchDims) n *= d;
        return n;
    }

    /// <summary>Fills per-batch storage offsets for three operands sharing batchDims.</summary>
    static void FillBatchOffsets(ReadOnlySpan<int> batchDims, int[] xSteps, int[] ySteps, int[] zSteps, int[] xOff, int[] yOff, int[] zOff)
    {
        int r = batchDims.Length;
        var coords = new int[r];
        int ox = 0, oy = 0, oz = 0;
        for (int b = 0; b < xOff.Length; b++)
        {
            xOff[b] = ox; yOff[b] = oy; zOff[b] = oz;
            for (int d = r - 1; d >= 0; d--)
            {
                coords[d]++;
                ox += xSteps[d]; oy += ySteps[d]; oz += zSteps[d];
                if (coords[d] < batchDims[d]) break;
                coords[d] = 0;
                ox -= xSteps[d] * batchDims[d]; oy -= ySteps[d] * batchDims[d]; oz -= zSteps[d] * batchDims[d];
            }
        }
    }

    static int[] BatchStrides(Tensor<float> t) =>
        t is BroadcastedTensor<float> b && b.effectiveStrides is not null ? b.effectiveStrides : t.strides;

    /// <summary>
    /// Batch-operand preparation: dense tensors pass through as before, and
    /// broadcast views over dense row-major matrix storage pass through with
    /// zero batch strides instead of being materialized per batch entry.
    /// Every other layout takes the exact previous copy path.
    /// </summary>
    static Tensor<float> RequireBatchOperand(Tensor<float> t, string name, ICopyAccountant? copy)
    {
        if (t is BroadcastedTensor<float> b && HasDenseMatrixCore(b)) return t;
        return RequireContiguous(t, name, copy);
    }

    static bool HasDenseMatrixCore(BroadcastedTensor<float> b)
    {
        int r = b.Rank;
        if (r < 2) return false;
        var dims = b.Dimensions;
        var eff = b.effectiveStrides;
        if (eff is null || eff.Length != r) return false;
        if (eff[r - 1] != 1 || eff[r - 2] != dims[r - 1]) return false;
        try
        {
            var unused = b.Storage;
            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }

    /// <summary>
    /// Folds a shared-right batch into one matrix product when every batch
    /// reads the same right block and the left/output batch blocks tile
    /// their storage densely. With batchCount batches of [m,n] over one
    /// [n,k], a single [batchCount*m,n] by [n,k] product computes identical
    /// dot products in the same order, so results match the per-batch loop
    /// up to the vectorized kernel the shared dispatcher selects per width.
    /// Size-one batch dims accept any step since they contribute one block
    /// at offset zero. Broadcast, strided, or truly batched right operands
    /// keep the per-batch fallback. The folded destination aliases the same
    /// output storage, preserving shape. Panel-packed kernels overwrite it;
    /// unpacked fallback lanes clear it first.
    /// </summary>
    static bool TryFoldSharedRightBatch(Tensor<float> bx, Tensor<float> by, Tensor<float> z, int[] batchDims, int[] xSteps, int[] ySteps, int[] zSteps, int batchCount, int m, int n, int k, TensorExecutionOptions options)
    {
        if (batchDims.Length == 0 || batchCount < 2) return false;
        foreach (var s in ySteps) if (s != 0) return false;
        if (!IsDenseBatchLayout(batchDims, xSteps, (long)m * n)) return false;
        if (!IsDenseBatchLayout(batchDims, zSteps, (long)m * k)) return false;
        int fm = checked(batchCount * m);
        var fx = new DenseTensor<float>(bx.Storage.Slice(0, checked(fm * n)), new[] { fm, n });
        var fy = new DenseTensor<float>(by.Storage.Slice(0, checked(n * k)), new[] { n, k });
        var fz = new DenseTensor<float>(z.Storage.Slice(0, checked(fm * k)), new[] { fm, k });
        MatMul2DCore(fx, fy, fz, options, overwriteDestination: true);
        return true;
    }

    /// <summary>
    /// True when odometer-order batch blocks of the given element count tile
    /// storage contiguously from offset zero. Size-one dims are exempt: their
    /// single block sits at the accumulated offset whatever the step.
    /// </summary>
    static bool IsDenseBatchLayout(int[] batchDims, int[] steps, long block)
    {
        long stride = block;
        for (int d = batchDims.Length - 1; d >= 0; d--)
        {
            if (batchDims[d] != 1 && steps[d] != stride) return false;
            stride *= batchDims[d];
        }
        return true;
    }

    // Few calls per run never trip Tier0 promotion counters; force Tier1 (see PLAN qdA2).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static void RunBatchedFloatMatMul(Tensor<float> bx, Tensor<float> by, Tensor<float> z, TensorExecutionOptions options, bool overwriteDestination)
    {
        bx = RequireBatchOperand(bx, nameof(bx), options.CopyReporter);
        by = RequireBatchOperand(by, nameof(by), options.CopyReporter);
        z = RequireContiguous(z, nameof(z), options.CopyReporter);
        var batchDims = bx.Dimensions[0..^2].ToArray();
        var m = bx.Dimensions[^2];
        var n = bx.Dimensions[^1];
        var k = by.Dimensions[^1];
        var xSteps = BatchSteps(batchDims, bx.Dimensions, BatchStrides(bx));
        var ySteps = BatchSteps(batchDims, by.Dimensions, BatchStrides(by));
        var zSteps = BatchSteps(batchDims, z.Dimensions, z.strides);
        int batchCount = BatchCount(batchDims);
        int dop = options.MaxDegreeOfParallelism < 2 || batchCount < 2
            ? 1
            : Math.Min(options.MaxDegreeOfParallelism, batchCount);
        if (TryFoldSharedRightBatch(bx, by, z, batchDims, xSteps, ySteps, zSteps, batchCount, m, n, k, options))
        {
            return;
        }
        if (ResolvePackedKernel(options, by, m) is { } packedB && (batchCount == 1 || ySteps.All(s => s == 0)))
        {
            RunPackedBatches(bx, z, batchDims, xSteps, zSteps, batchCount, dop, m, n, k, packedB, overwrite: true);
            return;
        }
        using var xh = bx.Storage.Pin();
        using var yh = by.Storage.Pin();
        using var zh = z.Storage.Pin();
        IntPtr xp0, yp0, zp0;
        unsafe { xp0 = (IntPtr)xh.Pointer; yp0 = (IntPtr)yh.Pointer; zp0 = (IntPtr)zh.Pointer; }
        if (dop > 1)
        {
            var xOff = new int[batchCount];
            var yOff = new int[batchCount];
            var zOff = new int[batchCount];
            FillBatchOffsets(batchDims, xSteps, ySteps, zSteps, xOff, yOff, zOff);
            Parallel.For(0, batchCount, new ParallelOptions { MaxDegreeOfParallelism = dop }, bi =>
            {
                unsafe
                {
                    RunFloatMatMulKernel(m, n, k,
                        (float*)xp0 + xOff[bi],
                        (float*)yp0 + yOff[bi],
                        (float*)zp0 + zOff[bi], options, overwriteDestination);
                }
            });
        }
        else
        {
            unsafe
            {
                var xp = (float*)xp0;
                var yp = (float*)yp0;
                var zp = (float*)zp0;
                int r = batchDims.Length;
                var coords = new int[r];
                int ox = 0, oy = 0, oz = 0;
                for (int b = 0; b < batchCount; b++)
                {
                    RunFloatMatMulKernel(m, n, k, xp + ox, yp + oy, zp + oz, options, overwriteDestination);
                    for (int d = r - 1; d >= 0; d--)
                    {
                        coords[d]++;
                        ox += xSteps[d]; oy += ySteps[d]; oz += zSteps[d];
                        if (coords[d] < batchDims[d]) break;
                        coords[d] = 0;
                        ox -= xSteps[d] * batchDims[d]; oy -= ySteps[d] * batchDims[d]; oz -= zSteps[d] * batchDims[d];
                    }
                }
            }
        }
    }

    /// <summary>
    /// Batched float MatMul reading B from a panel-packed clone shared by
    /// every batch. Mirrors RunBatchedFloatMatMul loop-for-loop with the
    /// packed kernel; the shared buffer pins once outside the loops.
    /// </summary>
    /// <summary>
    /// Runs one panel-packed product as 12/6-row AVX512 heads with 3/2-row
    /// tails: exact shapes keep single calls bit-identically, while larger
    /// counts trade 2-row passes for 6-way panel sharing. Rows are
    /// independent, so partitioning never changes per-element arithmetic;
    /// every piece uses the same panel layout and FMA order. Odd remainders
    /// peel one 3-row head, leaving an even tail for one 2-row call, so no
    /// scalar tail or original-operand read ever appears here.
    /// </summary>
    /// <summary>
    /// Runs one panel-packed product as 12-row AVX512 heads, 8-row groups
    /// absorbing 2/4-row remainders, one 6-row head, and 3/2-row tails: exact
    /// shapes keep single calls bit-identically, while larger counts trade
    /// narrow-tail panel re-reads for wider sharing. Rows are independent, so
    /// partitioning never changes per-element arithmetic; every piece uses the
    /// same panel layout and FMA order. Odd remainders peel one 3-row head,
    /// leaving an even tail for one 2-row call, so no scalar tail or
    /// original-operand read ever appears here. Callers admit m >= 2 through
    /// the packed gate.
    /// </summary>
    /// <summary>
    /// Minimum packed-panel bytes for the tile-major composer: below an L2-resident
    /// panel every grouping re-reads from fast cache anyway, so the legacy grouped
    /// calls stay.
    /// </summary>
    const long TiledComposerMinPanelBytes = 1024L * 1024;

    /// <summary>
    /// Tile-major packed row-group composer (M4): chains every row group per panel
    /// tile instead of sweeping the panel once per group call. Fires only where the
    /// legacy path emits at least two full-tile sweeps and every group has a tile
    /// entry (12/8/3/2-wide rows with no narrower remainder and no column remnant
    /// for narrow groups); single-sweep shapes keep the legacy path. Per-element
    /// arithmetic and order match bit for bit.
    /// </summary>
    /// <returns>True when the product was computed and the caller must return.</returns>
    // Few calls per run never trip Tier0 promotion counters; force Tier1 (see PLAN qdA2).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static unsafe bool TryRunPackedRowGroupsTiled(int m, int n, int k, float* x, float* packed, float* dest, bool overwrite)
    {
        if (m < 2) return false;
        if (!Avx512F.IsSupported || !Fma.IsSupported) return false;
        if ((long)n * k * sizeof(float) <= TiledComposerMinPanelBytes) return false;
        int main = (m / 12) * 12;
        int rem = m - main;
        if (rem == 1) { main -= 12; rem = 13; }
        if (rem == 4 && main >= 12) { main -= 12; rem = 16; }
        else if (rem == 2 && main >= 12) { main -= 12; rem = 14; }
        int rest = rem;
        int eightTotal = 0;
        while (rest >= 8 && rest != 9) { eightTotal += 8; rest -= 8; }
        int narrow3 = 0;
        int narrow2 = 0;
        int narrowSweeps = 0;
        switch (rest)
        {
            case 0: break;
            case 2: narrow2 = 2; narrowSweeps = 1; break;
            case 3: narrow3 = 3; narrowSweeps = 1; break;
            case 4: narrow2 = 4; narrowSweeps = 1; break;
            case 5: narrow3 = 3; narrow2 = 2; narrowSweeps = 2; break;
            case 7: narrow3 = 3; narrow2 = 4; narrowSweeps = 2; break;
            default: return false;
        }
        int legacySweeps = (main > 0 ? 1 : 0) + (eightTotal > 0 ? 1 : 0) + narrowSweeps;
        if (legacySweeps < 2) return false;
        int tileStep = 2 * Vector512<float>.Count;
        int blocked = k - (k % tileStep);
        if ((narrow3 > 0 || narrow2 > 0) && k != blocked) return false;
        int tiles = blocked / tileStep;
        int groups12 = main / 12;
        for (int tb = 0; tb < tiles; tb++)
        {
            int kb = tb * tileStep;
            float* panel = packed + tb * n * tileStep;
            float* xr = x;
            float* dr = dest;
            for (int g = 0; g < groups12; g++)
            {
                mm_avx512_12x32packed_tile(12, n, xr, panel, dr, k, kb, overwrite);
                xr += 12 * n;
                dr += 12 * k;
            }
            if (eightTotal > 0)
            {
                mm_avx512_8x32packed_tile(eightTotal, n, xr, panel, dr, k, kb, overwrite);
                xr += eightTotal * n;
                dr += eightTotal * k;
            }
            if (narrow3 > 0)
            {
                mm_3x4packed_tile(narrow3, n, xr, panel, dr, k, kb, overwrite);
                xr += narrow3 * n;
                dr += narrow3 * k;
            }
            if (narrow2 > 0)
            {
                mm_2x4packed_tile(narrow2, n, xr, panel, dr, k, kb, overwrite);
                xr += narrow2 * n;
                dr += narrow2 * k;
            }
        }
        int remCols = k - blocked;
        if (remCols > 0)
        {
            float* xr = x;
            float* dr = dest;
            for (int g = 0; g < groups12; g++)
            {
                mm_avx512_12x32packed_col_tail(12, n, k, xr, packed, dr, blocked, tiles, remCols, overwrite);
                xr += 12 * n;
                dr += 12 * k;
            }
            if (eightTotal > 0)
            {
                mm_avx512_8x32packed_col_tail(eightTotal, n, k, xr, packed, dr, blocked, tiles, remCols, overwrite);
            }
        }
        return true;
    }



    // Few calls per run never trip Tier0 promotion counters; force Tier1 (see PLAN qdA2).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static unsafe void RunPackedRowGroups(int m, int n, int k, float* x, float* packed, float* dest, bool overwrite)
    {
        int rest = m;
        float* xr = x;
        float* dr = dest;
            if (TryRunPackedRowGroupsTiled(m, n, k, x, packed, dest, overwrite)) return;
        if (Avx512F.IsSupported && rest >= 6)
        {
            int main = (rest / 12) * 12;
            int rem = rest - main;
            if (rem == 1) { main -= 12; rem = 13; }
            // Absorb narrow remainders into 8-row groups: a 4-row 2-row tail
            // re-reads every panel twice at 2-way sharing (measured ~half the
            // time on M=16 encoder shapes), so trade one 12-row head for 8+8
            // (rem 4) or 8+6 (rem 2) instead. Rows stay independent and every
            // piece shares panel layout and FMA order, so this stays bitwise.
            if (rem == 4 && main >= 12) { main -= 12; rem = 16; }
            else if (rem == 2 && main >= 12) { main -= 12; rem = 14; }
            if (main > 0)
            {
                mm_unsafe_vectorized_avx512_12x32packed(main, n, k, xr, packed, dr, overwrite);
                xr += main * n;
                dr += main * k;
            }
            rest = rem;
            // Never leave a single row: rem 9 drains to 6+3 through the peel
            // below instead of 8+1, which no kernel covers.
            // The 8-row kernel loops its groups tile-major inside one call, so emit
            // the whole 8-row region as a single call: separate M=8 calls would
            // re-sweep the full packed panel once per call, while one call streams
            // each tile once for all its groups. Grouping decisions are unchanged,
            // so per-element arithmetic and order match bit for bit.
            int eightTotal = 0;
            while (rest >= 8 && rest != 9)
            {
                eightTotal += 8;
                rest -= 8;
            }
            if (eightTotal > 0)
            {
                mm_unsafe_vectorized_avx512_8x32packed(eightTotal, n, k, xr, packed, dr, overwrite);
                xr += eightTotal * n;
                dr += eightTotal * k;
            }
            if (rest >= 6 && rest != 7)
            {
                mm_unsafe_vectorized_avx512_6x32packed(6, n, k, xr, packed, dr, overwrite);
                xr += 6 * n;
                dr += 6 * k;
                rest -= 6;
            }
        }
        if (rest == 0)
        {
            return;
        }
        if ((rest % 3) == 0)
        {
            mm_unsafe_vectorized_intrinsics_3x4packed(rest, n, k, xr, packed, dr, overwrite);
            return;
        }
        if ((rest & 1) == 0)
        {
            mm_unsafe_vectorized_intrinsics_2x4packed(rest, n, k, xr, packed, dr, overwrite);
            return;
        }
        mm_unsafe_vectorized_intrinsics_3x4packed(3, n, k, xr, packed, dr, overwrite);
        mm_unsafe_vectorized_intrinsics_2x4packed(rest - 3, n, k, xr + 3 * n, packed, dr + 3 * k, overwrite);
    }

    // Tier0-stuck leaf (see PLAN qdA2): force Tier1; few benchmark calls never trip promotion.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static void RunPackedBatches(Tensor<float> bx, Tensor<float> z, int[] batchDims, int[] xSteps, int[] zSteps, int batchCount, int dop, int m, int n, int k, DenseTensor<float> packed, bool overwrite)
    {
        using var xh = bx.Storage.Pin();
        using var ph = packed.Buffer.Pin();
        using var zh = z.Storage.Pin();
        IntPtr xp0, zp0;
        unsafe { xp0 = (IntPtr)xh.Pointer; zp0 = (IntPtr)zh.Pointer; }
        unsafe
        {
            float* pp = (float*)ph.Pointer;
            if (dop > 1)
            {
                var xOff = new int[batchCount];
                var zOff = new int[batchCount];
                FillBatchOffsets(batchDims, xSteps, zSteps, zSteps, xOff, new int[batchCount], zOff);
                Parallel.For(0, batchCount, new ParallelOptions { MaxDegreeOfParallelism = dop }, bi =>
                {
                    unsafe
                    {
                        // Row groups with 12/6-row AVX512 heads and 3/2-row tails (P5).
                        RunPackedRowGroups(m, n, k,
                            (float*)xp0 + xOff[bi],
                            pp,
                            (float*)zp0 + zOff[bi], overwrite);
                    }
                });
            }
            else
            {
                var xp = (float*)xp0;
                var zp = (float*)zp0;
                int r = batchDims.Length;
                var coords = new int[r];
                int ox = 0, oz = 0;
                for (int b = 0; b < batchCount; b++)
                {
                    RunPackedRowGroups(m, n, k, xp + ox, pp, zp + oz, overwrite);
                    for (int d = r - 1; d >= 0; d--)
                    {
                        coords[d]++;
                        ox += xSteps[d]; oz += zSteps[d];
                        if (coords[d] < batchDims[d]) break;
                        coords[d] = 0;
                        ox -= xSteps[d] * batchDims[d]; oz -= zSteps[d] * batchDims[d];
                    }
                }
            }
        }
    }
    /// <summary>
    /// Writes the float matrix product into an existing dense destination,
    /// overwriting it. The destination must not alias either input.
    /// Panel-packed kernels overwrite it; unpacked fallback lanes clear it first.
    /// </summary>
    public static Tensor<float> MatMul(Tensor<float> x, Tensor<float> y, DenseTensor<float> destination, TensorExecutionOptions options)

    {
        return MatMulInto(x, y, destination, options, overwriteDestination: true);
    }

    // Few calls per run never trip Tier0 promotion counters; force Tier1 (see PLAN qdA2).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static Tensor<float> MatMulInto(Tensor<float> x, Tensor<float> y, DenseTensor<float> destination, TensorExecutionOptions options, bool overwriteDestination)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        if (ReferenceEquals(destination, x) || ReferenceEquals(destination, y) || TensorAlias.SharesBackingMemory(destination, x) || TensorAlias.SharesBackingMemory(destination, y)) throw new ArgumentException(nameof(destination), "Destination must not alias the input tensors.");
        var plan = MatMulShapes.Create(x.Dimensions, y.Dimensions);
        if (!destination.Dimensions.SequenceEqual(plan.OutputShape)) throw new ArgumentException(nameof(destination), "Destination shape must match the matrix product shape.");
        if (!HasStandardStrides(destination)) throw new ArgumentException(nameof(destination), "Destination must have standard row-major strides.");
        var px = plan.PromoteX ? x.InsertDim(0) : x;
        var py = plan.PromoteY ? y.InsertDim(y.Rank) : y;
        if (px.Rank == 2 && py.Rank == 2)
        {
            var dd = destination.dimensions;
            var destView = dd.Length == 2 && dd[0] == px.dimensions[0] && dd[1] == py.dimensions[1]
                ? destination
                : new DenseTensor<float>(destination.Buffer, new int[] { px.dimensions[0], py.dimensions[1] });
            MatMul2DCore(px, py, destView, options, overwriteDestination);
            return destination;
        }
        var xdl = px.Dimensions[^2..];
        var ydl = py.Dimensions[^2..];
        if (xdl[1] != ydl[0])
        {
            throw new ArgumentException($"The number of columns in the first matrix ({xdl[1]}) is not equal to the number of rows in the second matrix ({ydl[0]}).");
        }
        StartOpStage(OpStage.Broadcast);
        if (!BroadcastShape(px.Dimensions[0..^2], py.Dimensions[0..^2], out var bd))
        {
            throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
        }

        var bdx = bd.Append(xdl[0]).Append(xdl[1]).ToArray();
        if (!Tensor<float>.Broadcast(px, bdx, out var bx))
        {
            throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
        }
        var bdy = bd.Append(ydl[0]).Append(ydl[1]).ToArray();
        if (!Tensor<float>.Broadcast(py, bdy, out var by))
        {
            throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
        }

        StartOpStage(OpStage.Math);
        var coreDims = bd.Append(xdl[0]).Append(ydl[1]).ToArray();
        var target = coreDims.SequenceEqual(destination.dimensions) ? destination : new DenseTensor<float>(destination.Buffer, coreDims);
        RunBatchedFloatMatMul(bx, by, target, options, overwriteDestination);
        return destination;
    }


    /// <summary>Computes the float matrix product, renting the output from the pool when provided.</summary>
    public static Tensor<float> MatMul(Tensor<float> x, Tensor<float> y, TensorExecutionOptions options, TensorBufferPool? pool)
    {
        if (pool is null) return MatMul(x, y, options);
        var dims = MatMulOutputShape(x.Dimensions, y.Dimensions);
        long length = 1;
        checked
        {
            foreach (var d in dims) length *= d;
        }
        if (length > int.MaxValue) throw new ArgumentException("MatMul output element count exceeds maximum backing-store length.");
        var destination = new DenseTensor<float>(new Memory<float>(pool.Rent<float>((int)length)), dims);
        return MatMulInto(x, y, destination, options, overwriteDestination: true);
    }
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]  
    public static Tensor<float> MatMul(Tensor<float> x, Tensor<float> y) => MatMul(x, y, TensorExecutionOptions.Auto);

    public static Tensor<float> MatMul(Tensor<float> x, Tensor<float> y, TensorExecutionOptions options)
    
    {
        var (plan, px, py) = PlanMatMul(x, y);
        Tensor<float> core;
        if (px.Rank == 2 && py.Rank == 2)
        {
            core = Tensor<float>.MatMul2D(px, py, options);
        }
        else
        {
            var xdl = px.Dimensions[^2..];
            var ydl = py.Dimensions[^2..];
            if (xdl[1] != ydl[0])
            {
                throw new ArgumentException($"The number of columns in the first matrix ({xdl[1]}) is not equal to the number of rows in the second matrix ({ydl[0]}).");
            }
            StartOpStage(OpStage.Broadcast);
            if (!BroadcastShape(px.Dimensions[0..^2], py.Dimensions[0..^2], out var bd))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }

            var bdx = bd.Append(xdl[0]).Append(xdl[1]).ToArray();
            if (!Tensor<float>.Broadcast(px, bdx, out var bx))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }
            var bdy = bd.Append(ydl[0]).Append(ydl[1]).ToArray();
            if (!Tensor<float>.Broadcast(py, bdy, out var by))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }

            StartOpStage(OpStage.Math);

            var z = DenseTensor<float>.OfShape(bd.Append(xdl[0]).Append(ydl[1]).ToArray());
            RunBatchedFloatMatMul(bx, by, z, options, overwriteDestination: false);
            core = z;
        }
        return MatMulShapes.Squeeze(core, plan);
    }


    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static Tensor<double> MatMul(Tensor<double> x, Tensor<double> y) => MatMul(x, y, TensorExecutionOptions.Auto);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static Tensor<double> MatMul(Tensor<double> x, Tensor<double> y, TensorExecutionOptions options)
    
    {
        var (plan, px, py) = PlanMatMul(x, y);
        Tensor<double> core;
        if (px.Rank == 2 && py.Rank == 2)
        {
            core = Tensor<double>.MatMul2D(px, py, options);
        }
        else
        {
            var (bx, by, z, batchDims, xSteps, ySteps, zSteps, batchCount, m, n, k) = PlanBatchedMatMul(px, py, options.CopyReporter);

            StartOpStage(OpStage.Math);
            using var xh = bx.Storage.Pin();
            using var yh = by.Storage.Pin();
            using var zh = z.Storage.Pin();
            unsafe
            {
                var xp = (double*)xh.Pointer;
                var yp = (double*)yh.Pointer;
                var zp = (double*)zh.Pointer;
                int r = batchDims.Length;
                var coords = new int[r];
                int ox = 0, oy = 0, oz = 0;
                for (int b = 0; b < batchCount; b++)
                {
                    if (options.UseSimd && options.UseIntrinsics && Fma.IsSupported)
                    {
                        mm_unsafe_vectorized_intrinsics(m, n, k, xp + ox, yp + oy, zp + oz);
                    }
                    else if (options.UseSimd)
                    {
                        mm_unsafe_vectorized(m, n, k, xp + ox, yp + oy, zp + oz);
                    }
                    else
                    {
                        mm(m, n, k, xp + ox, yp + oy, zp + oz);
                    }
                    for (int d = r - 1; d >= 0; d--)
                    {
                        coords[d]++;
                        ox += xSteps[d]; oy += ySteps[d]; oz += zSteps[d];
                        if (coords[d] < batchDims[d]) break;
                        coords[d] = 0;
                        ox -= xSteps[d] * batchDims[d]; oy -= ySteps[d] * batchDims[d]; oz -= zSteps[d] * batchDims[d];
                    }
                }
            }
            core = z;
        }
        return MatMulShapes.Squeeze(core, plan);
    }

    static bool HasStandardStrides<TElement>(DenseTensor<TElement> tensor) where TElement : unmanaged
    {
        if (tensor.IsReversedStride) return false;
        return tensor.strides.SequenceEqual(ArrayUtilities.GetStrides(tensor.dimensions));
    }

    internal static DenseTensor<TElement> RequireContiguous<TElement>(Tensor<TElement> t, string name, ICopyAccountant? copy) where TElement : unmanaged
    {
        if (t is DenseTensor<TElement> d && !d.IsReversedStride && HasStandardStrides(d))
        {
            if (d.Buffer.Length != (int)d.Length) throw new ArgumentException(name + " backing length does not match shape.");
            return d;
        }
        return CountedCopy(t.ToDenseTensor(), copy);
    }

    internal static DenseTensor<TElement> CountedCopy<TElement>(DenseTensor<TElement> dense, ICopyAccountant? copy) where TElement : unmanaged
    {
        copy?.AddCopyBytes((long)dense.Length * Unsafe.SizeOf<TElement>());
        return dense;
    }
}

internal static class MatMulShapes
{
    public readonly struct Plan
    {
        public readonly bool PromoteX;
        public readonly bool PromoteY;
        public readonly int[] OutputShape;
        public Plan(bool promoteX, bool promoteY, int[] outputShape)
        {
            PromoteX = promoteX;
            PromoteY = promoteY;
            OutputShape = outputShape;
        }
    }

    static int[] CoreOutputShape(System.ReadOnlySpan<int> xd, System.ReadOnlySpan<int> yd)
    {
        if (xd.Length == 2 && yd.Length == 2)
        {
            if (xd[1] != yd[0]) throw new System.ArgumentException($"The number of columns in the first matrix ({xd[1]}) is not equal to the number of rows in the second matrix ({yd[0]}).");
            return new int[] { xd[0], yd[1] };
        }
        var xdl = xd[^2..];
        var ydl = yd[^2..];
        if (xdl[1] != ydl[0]) throw new System.ArgumentException($"The number of columns in the first matrix ({xdl[1]}) is not equal to the number of rows in the second matrix ({ydl[0]}).");
        if (!Tensor<int>.BroadcastShape(xd[0..^2], yd[0..^2], out var bd)) throw new System.ArgumentException("The tensor shapes are not compatible for broadcasting.");
        return bd.Append(xdl[0]).Append(ydl[1]).ToArray();
    }

    public static Plan Create(System.ReadOnlySpan<int> xd, System.ReadOnlySpan<int> yd)
    {
        if (xd.Length == 0 || yd.Length == 0) throw new System.ArgumentException("The rank of each tensor in matrix multiplication must be at least 1; rank-1 vectors are promoted.");
        bool promoteX = xd.Length == 1;
        bool promoteY = yd.Length == 1;
        int[] px = promoteX ? new int[] { 1, xd[0] } : xd.ToArray();
        int[] py = promoteY ? new int[] { yd[0], 1 } : yd.ToArray();
        int[] core = CoreOutputShape(px, py);
        int[] output;
        if (promoteX && promoteY) output = core[0..^2];
        else if (promoteX) output = core[0..^2].Append(core[^1]).ToArray();
        else if (promoteY) output = core[0..^1];
        else output = core;
        return new Plan(promoteX, promoteY, output);
    }

    public static Tensor<U> Squeeze<U>(Tensor<U> core, Plan plan) where U : unmanaged
    {
        if (plan.PromoteX && plan.PromoteY)
        {
            var once = core.RemoveDim(core.Rank - 2);
            return once.RemoveDim(once.Rank - 1);
        }
        if (plan.PromoteX) return core.RemoveDim(core.Rank - 2);
        if (plan.PromoteY) return core.RemoveDim(core.Rank - 1);
        return core;
    }
}