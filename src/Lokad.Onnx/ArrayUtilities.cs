namespace Lokad.Onnx
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Runtime.CompilerServices;

    public static class ArrayUtilities
    {
        public const int StackallocMax = 16;

        /// <summary>
        /// Copies runs of elements along one axis for every outer block.
        /// Block o reads length*inner elements at ((o * srcAxisLength) + srcStart) * inner
        /// and writes them at ((o * dstAxisLength) + dstStart) * inner.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static void CopyAxisChunks<T>(ReadOnlySpan<T> src, int srcAxisLength, Span<T> dst, int dstAxisLength, int outer, int inner, int srcStart, int dstStart, int length)
        {
            for (int o = 0; o < outer; o++)
            {
                int from = (o * srcAxisLength + srcStart) * inner;
                int to = (o * dstAxisLength + dstStart) * inner;
                src.Slice(from, length * inner).CopyTo(dst.Slice(to, length * inner));
            }
        }

        /// <summary>
        /// Multiplies dimensions[startIndex..] with overflow checking.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static int ComputeOffsetForReduction(ReadOnlySpan<int> dimensions, int startIndex)
        {
            if (startIndex < 0 || startIndex > dimensions.Length) throw new ArgumentOutOfRangeException(nameof(startIndex));
            try
            {
                checked
                {
                    int product = 1;
                    for (int i = startIndex; i < dimensions.Length; i++)
                    {
                        if (dimensions[i] < 0) throw new ArgumentException("Dimensions must be non-negative.", nameof(dimensions));
                        product *= dimensions[i];
                    }
                    return product;
                }
            }
            catch (OverflowException ex) { throw new ArgumentException("Tensor shape product overflows.", nameof(dimensions), ex); }
        }

        /// <summary>
        /// Multiplies dimensions[startIndex..] with overflow checking into a long.
        /// Unlike the int version, an overflowing product escapes as OverflowException.
        /// </summary>
        public static long ComputeOffsetForReductionLong(ReadOnlySpan<int> dimensions, int startIndex)
        {
            if (startIndex < 0 || startIndex > dimensions.Length) throw new ArgumentOutOfRangeException(nameof(startIndex));
            checked
            {
                long product = 1;
                for (int i = startIndex; i < dimensions.Length; i++)
                {
                    if (dimensions[i] < 0) throw new ArgumentException("Dimensions must be non-negative.", nameof(dimensions));
                    product *= dimensions[i];
                }
                return product;
            }
        }

        public static bool IsAscending(ReadOnlySpan<int> values)
        {
            for (int i = 1; i < values.Length; i++)
            {
                if (values[i] < values[i - 1]) return false;
            }
            return true;
        }

        public static bool IsDescending(ReadOnlySpan<int> values)
        {
            for (int i = 1; i < values.Length; i++)
            {
                if (values[i] > values[i - 1]) return false;
            }
            return true;
        }

        /// <summary>
        /// Row-major strides for the given dimensions: the last dimension has stride 1.
        /// </summary>
        public static int[] GetStrides(ReadOnlySpan<int> dimensions) => GetStrides(dimensions, false);

        /// <summary>
        /// Strides for the given dimensions, row-major by default or column-major
        /// (first dimension stride 1) when reverseStride holds. Rejects negative
        /// dimensions and overflowing products.
        /// </summary>
        public static int[] GetStrides(ReadOnlySpan<int> dimensions, bool reverseStride)
        {
            if (dimensions.Length == 0) return Array.Empty<int>();
            try
            {
                checked
                {
                    foreach (var d in dimensions)
                    {
                        if (d < 0) throw new ArgumentException("Dimensions must be non-negative.", nameof(dimensions));
                    }
                    var strides = new int[dimensions.Length];
                    int stride = 1;
                    if (reverseStride)
                    {
                        for (int i = 0; i < strides.Length; i++)
                        {
                            strides[i] = stride;
                            stride *= dimensions[i];
                        }
                    }
                    else
                    {
                        for (int i = strides.Length - 1; i >= 0; i--)
                        {
                            strides[i] = stride;
                            stride *= dimensions[i];
                        }
                    }
                    return strides;
                }
            }
            catch (OverflowException ex) { throw new ArgumentException("Tensor shape product overflows.", nameof(dimensions), ex); }
        }

        /// <summary>
        /// Partitions strides by axis membership, preserving order: entries whose
        /// axis is listed go to splitStrides, the rest to newStrides.
        /// </summary>
        public static void SplitStrides(int[] strides, int[] splitAxes, int[] newStrides, int stridesOffset, int[] splitStrides, int splitStridesOffset)
        {
            int kept = 0;
            for (int i = 0; i < strides.Length; i++)
            {
                bool split = false;
                for (int j = 0; j < splitAxes.Length; j++)
                {
                    if (splitAxes[j] == i)
                    {
                        splitStrides[splitStridesOffset + j] = strides[i];
                        split = true;
                        break;
                    }
                }
                if (!split)
                {
                    newStrides[stridesOffset + kept] = strides[i];
                    kept++;
                }
            }
        }

        /// <summary>
        /// Dot product of strides and indices: the flat offset of n-d indices.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static int GetIndex(int[] strides, ReadOnlySpan<int> indices) => GetIndex(strides, indices, 0);

        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static int GetIndex(int[] strides, ReadOnlySpan<int> indices, int startFromDimension)
        {
            if (strides.Length == 0) return 0;
            checked
            {
                int index = 0;
                for (int i = startFromDimension; i < indices.Length; i++) index += strides[i] * indices[i];
                return index;
            }
        }

        /// <summary>
        /// Flat offset skipping broadcast dimensions and zero entries.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static int GetIndex(int[] strides, ReadOnlySpan<int> indices, int[] broadcastedDims, int startFromDimension)
        {
            Debug.Assert(strides.Length == indices.Length);
            int index = 0;
            for (int i = startFromDimension; i < indices.Length; i++)
            {
                if (indices[i] != 0 && Array.IndexOf(broadcastedDims, i) == -1) index += strides[i] * indices[i];
            }
            return index;
        }

        /// <summary>
        /// Decomposes a flat index into n-d indices, dividing by the largest
        /// stride first for reverse layouts.
        /// </summary>
        public static void GetIndices(ReadOnlySpan<int> strides, bool reverseStride, int index, int[] indices, int startFromDimension)
        {
            Debug.Assert(reverseStride ? IsAscending(strides) : IsDescending(strides), "Index decomposition requires ordered strides");
            Debug.Assert(strides.Length == indices.Length);
            if (indices.Length == 0) return;
            int remainder = index;
            for (int i = startFromDimension; i < strides.Length; i++)
            {
                int axis = reverseStride ? strides.Length - 1 - i : i;
                indices[axis] = remainder / strides[axis];
                remainder %= strides[axis];
            }
        }

        /// <summary>
        /// Decomposes a flat index into n-d indices, dividing by the largest
        /// stride first for reverse layouts.
        /// </summary>
        public static void GetIndices(ReadOnlySpan<int> strides, bool reverseStride, int index, Span<int> indices, int startFromDimension)
        {
            Debug.Assert(reverseStride ? IsAscending(strides) : IsDescending(strides), "Index decomposition requires ordered strides");
            Debug.Assert(strides.Length == indices.Length);
            if (indices.Length == 0) return;
            int remainder = index;
            for (int i = startFromDimension; i < strides.Length; i++)
            {
                int axis = reverseStride ? strides.Length - 1 - i : i;
                indices[axis] = remainder / strides[axis];
                remainder %= strides[axis];
            }
        }

        /// <summary>
        /// Re-expresses a flat index from one stride layout in another layout
        /// over the same coordinates.
        /// </summary>
        public static int TransformIndexByStrides(int index, int[] sourceStrides, bool sourceReverseStride, int[] transformStrides)
        {
            Debug.Assert(index >= 0);
            Debug.Assert(sourceReverseStride ? IsAscending(sourceStrides) : IsDescending(sourceStrides), "Index decomposition requires ordered strides");
            Debug.Assert(sourceStrides.Length == transformStrides.Length);
            if (sourceStrides.Length == 0)
            {
                Debug.Assert(index == 0, "Index has to be zero for a scalar tensor");
                return 0;
            }
            int mapped = 0;
            int remainder = index;
            for (int i = 0; i < sourceStrides.Length; i++)
            {
                int axis = sourceReverseStride ? sourceStrides.Length - 1 - i : i;
                mapped += transformStrides[axis] * (remainder / sourceStrides[axis]);
                remainder %= sourceStrides[axis];
            }
            return mapped;
        }

        /// <summary>
        /// Collects every leaf element of a nested array in depth-first order.
        /// </summary>
        public static T[] Flatten<T>(this Array data)
        {
            var flat = new List<T>();
            CollectLeaves<T>(data, flat);
            return flat.ToArray();
        }

        static void CollectLeaves<T>(Array node, List<T> flat)
        {
            foreach (var item in node)
            {
                if (item is Array nested) CollectLeaves<T>(nested, flat);
                else flat.Add((T)item);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static int HandleNegativeAxisOrIndex(int size, int axis) => axis >= 0 ? axis : size + axis;

        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static bool CheckNoRepeatedDims(int[] dims) => dims.Length == dims.Distinct().Count();

        public static int Clamp(int value, int min, int max)
        {
            if (value < min) return min;
            if (value > max) return max;
            return value;
        }

        public static int Clamp(int value, int pmin, int pmax, int nmin, int nmax)
        {
            int min = value >= 0 ? pmin : nmin;
            int max = value >= 0 ? pmax : nmax;
            if (value < min) return min;
            if (value > max) return max;
            return value;
        }

        /// <summary>
        /// Splits a shape into kept dimensions (in order) and reduced ones (in axes order).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static Tuple<int[], int[]> ComputeShapesForReduction(int[] inShape, int[] axes)
        {
            var kept = new List<int>(inShape.Length);
            for (int dim = 0; dim < inShape.Length; dim++)
            {
                if (Array.IndexOf(axes, dim) < 0) kept.Add(inShape[dim]);
            }
            var reduced = new int[axes.Length];
            for (int i = 0; i < axes.Length; i++) reduced[i] = inShape[axes[i]];
            return new Tuple<int[], int[]>(kept.ToArray(), reduced);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static int[] ComputeReducedShape(int[] inShape, int[] axes)
        {
            var reducedShape = new int[axes.Length];
            for (int i = 0; i < axes.Length; i++) reducedShape[i] = inShape[axes[i]];
            return reducedShape;
        }

        /// <summary>
        /// Permutation moving every non-reduced axis first (in order) and the
        /// reduced axes last, or null when they already sit innermost.
        /// </summary>
        public static int[]? GetAxesPermutationForReduction(int[] axes, int rank)
        {
            if (AxesAreInnerMostDims(axes, rank)) return null;
            var permutation = new List<int>(rank);
            for (int i = 0; i < rank; i++)
            {
                if (Array.IndexOf(axes, i) < 0) permutation.Add(i);
            }
            permutation.AddRange(axes);
            return permutation.ToArray();
        }

        public static bool AxesAreInnerMostDims(int[] axes, int rank)
        {
            for (int i = 0; i < axes.Length; i++)
            {
                if (axes[axes.Length - 1 - i] != rank - 1 - i) return false;
            }
            return true;
        }

        public static int[] GetInnerMostAxes(int n, int rank)
        {
            var axes = new int[n];
            for (int i = 0; i < n; i++) axes[i] = rank - n + i;
            return axes;
        }

        public static T[,] To2DArray<T>(this T[][] source)
        {
            var result = new T[source.Length, source[0].Length];
            for (int i = 0; i < source.Length; i++)
            {
                for (int j = 0; j < source[0].Length; j++) result[i, j] = source[i][j];
            }
            return result;
        }

        public static void UnsafeCopy<T>(T[] arr1, ref T[] arr2) where T : unmanaged
        {
            if (arr1.Length != arr2.Length) throw new ArgumentException("The arrays must be of the same length.");
            unsafe
            {
                Buffer.BlockCopy(arr1, 0, arr2, 0, arr1.Length * sizeof(T));
            }
        }
    }
}
