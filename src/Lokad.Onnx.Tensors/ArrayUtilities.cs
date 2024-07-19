// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is copied and adapted from the following git repository -
// https://github.com/dotnet/corefx
// Commit ID: bdd0814360d4c3a58860919f292a306242f27da1
// Path: /src/System.Numerics.Tensors/src/System/Numerics/Tensors/ArrayUtilities.cs
// Original license statement below -

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;

namespace Lokad.Onnx
{
    public static class ArrayUtilities
    {
        public const int StackallocMax = 16;

        public static int ComputeOffsetForReduction(ReadOnlySpan<int> dimensions, int startIndex = 0)
        {
            int product = 1;
            for (int i = startIndex; i < dimensions.Length; i++)
            {
                if (dimensions[i] < 0)
                {
                    throw new ArgumentOutOfRangeException($"{nameof(dimensions)}[{i}]");
                }

                // we use a long which should be much larger than is ever used here,
                // but still force checked
                checked
                {
                    product *= dimensions[i];
                }
            }

            return product;
        }

        public static bool IsAscending(ReadOnlySpan<int> values)
        {
            for (int i = 1; i < values.Length; i++)
            {
                if (values[i] < values[i - 1])
                {
                    return false;
                }
            }

            return true;
        }

        public static bool IsDescending(ReadOnlySpan<int> values)
        {
            for (int i = 1; i < values.Length; i++)
            {
                if (values[i] > values[i - 1])
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Gets the set of strides that can be used to calculate the offset of n-dimensions in a 1-dimensional layout
        /// </summary>
        /// <param name="dimensions"></param>
        /// <param name="reverseStride"></param>
        /// <returns></returns>
        public static int[] GetStrides(ReadOnlySpan<int> dimensions, bool reverseStride = false)
        {
            int[] strides = new int[dimensions.Length];

            if (dimensions.Length == 0)
            {
                return strides;
            }

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

        public static void SplitStrides(int[] strides, int[] splitAxes, int[] newStrides, int stridesOffset, int[] splitStrides, int splitStridesOffset)
        {
            int newStrideIndex = 0;
            for (int i = 0; i < strides.Length; i++)
            {
                int stride = strides[i];
                bool isSplit = false;
                for (int j = 0; j < splitAxes.Length; j++)
                {
                    if (splitAxes[j] == i)
                    {
                        splitStrides[splitStridesOffset + j] = stride;
                        isSplit = true;
                        break;
                    }
                }

                if (!isSplit)
                {
                    newStrides[stridesOffset + newStrideIndex++] = stride;
                }
            }
        }

        /// <summary>
        /// Calculates the 1-d index for n-d indices in layout specified by strides.
        /// </summary>
        /// <param name="strides"></param>
        /// <param name="indices"></param>
        /// <param name="startFromDimension"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static int GetIndex(int[] strides, ReadOnlySpan<int> indices, int startFromDimension = 0)
        {
            Debug.Assert(strides.Length == indices.Length);
            
            int index = 0;
            unchecked
            {
                for (int i = startFromDimension; i < indices.Length; i++)
                {
                    index += strides[i] * indices[i];
                }
            }
            return index;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public static int GetIndex(int[] strides, ReadOnlySpan<int> indices, int[] broadcastedDims, int startFromDimension = 0)
        {
            Debug.Assert(strides.Length == indices.Length);

            int index = 0;
            for (int i = startFromDimension; i < indices.Length; i++)
            {
                if (indices[i] == 0 || Array.IndexOf(broadcastedDims, i) != -1)
                {
                    continue;
                }
                else
                {
                    index += strides[i] * indices[i];
                }
            }

            return index;
        }
        /// <summary>
        /// Calculates the n-d indices from the 1-d index in a layout specified by strides
        /// </summary>
        /// <param name="strides"></param>
        /// <param name="reverseStride"></param>
        /// <param name="index"></param>
        /// <param name="indices"></param>
        /// <param name="startFromDimension"></param>
        public static void GetIndices(ReadOnlySpan<int> strides, bool reverseStride, int index, int[] indices, int startFromDimension = 0)
        {
            Debug.Assert(reverseStride ? IsAscending(strides) : IsDescending(strides), "Index decomposition requires ordered strides");
            Debug.Assert(strides.Length == indices.Length);

            // scalar tensor - nothing to process
            if (indices.Length == 0)
            {
                return;
            }

            int remainder = index;
            for (int i = startFromDimension; i < strides.Length; i++)
            {
                // reverse the index for reverseStride so that we divide by largest stride first
                var nIndex = reverseStride ? strides.Length - 1 - i : i;

                var stride = strides[nIndex];
                indices[nIndex] = remainder / stride;
                remainder %= stride;
            }
        }

        /// <summary>
        /// Calculates the n-d indices from the 1-d index in a layout specificed by strides
        /// </summary>
        /// <param name="strides"></param>
        /// <param name="reverseStride"></param>
        /// <param name="index"></param>
        /// <param name="indices"></param>
        /// <param name="startFromDimension"></param>
        public static void GetIndices(ReadOnlySpan<int> strides, bool reverseStride, int index, Span<int> indices, int startFromDimension = 0)
        {
            Debug.Assert(reverseStride ? IsAscending(strides) : IsDescending(strides), "Index decomposition requires ordered strides");
            Debug.Assert(strides.Length == indices.Length);

            // scalar tensor - nothing to process
            if (indices.Length == 0)
            {
                return;
            }

            int remainder = index;
            for (int i = startFromDimension; i < strides.Length; i++)
            {
                // reverse the index for reverseStride so that we divide by largest stride first
                var nIndex = reverseStride ? strides.Length - 1 - i : i;

                var stride = strides[nIndex];
                indices[nIndex] = remainder / stride;
                remainder %= stride;
            }
        }

        /// <summary>
        /// Takes an 1-d index over n-d sourceStrides and recalculates it assuming same n-d coordinates over a different n-d strides
        /// </summary>
        public static int TransformIndexByStrides(int index, int[] sourceStrides, bool sourceReverseStride, int[] transformStrides)
        {
            Debug.Assert(index >= 0);
            Debug.Assert(sourceReverseStride ? IsAscending(sourceStrides) : IsDescending(sourceStrides), "Index decomposition requires ordered strides");
            Debug.Assert(sourceStrides.Length == transformStrides.Length);

            // scalar tensor
            if (sourceStrides.Length == 0)
            {
                Debug.Assert(index == 0, "Index has to be zero for a scalar tensor");
                return 0;
            }

            int transformIndex = 0;
            int remainder = index;

            for (int i = 0; i < sourceStrides.Length; i++)
            {
                // reverse the index for reverseStride so that we divide by largest stride first
                var nIndex = sourceReverseStride ? sourceStrides.Length - 1 - i : i;

                var sourceStride = sourceStrides[nIndex];
                var transformStride = transformStrides[nIndex];

                transformIndex += transformStride * (remainder / sourceStride);
                remainder %= sourceStride;
            }

            return transformIndex;
        }

        public static T[] GetEmpty<T>()
        {
            // Match the implementation of Array.GetEmpty<T>()
            // from dotnet/runtime. Having it as a static in a
            // nested class ensures we only allocate the empty
            // array once and only when actually necessary.
            return EmptyArray<T>.Value;
        }
        private static class EmptyArray<T>
        {
            public static readonly T[] Value = new T[0];
        }

        public static T[] Flatten<T>(this Array data)
        {
            var list = new List<T>();
            var stack = new Stack<IEnumerator>();
            stack.Push(data.GetEnumerator());
            do
            {
                for (var iterator = stack.Pop(); iterator.MoveNext();)
                {
                    if (iterator.Current is Array)
                    {
                        stack.Push(iterator);
                        iterator = ((IEnumerable) iterator.Current).GetEnumerator();
                    }
                    else
                        list.Add((T)iterator.Current);
                }
            }
            while (stack.Count > 0);
            return list.ToArray();
        }

        public static int HandleNegativeAxisOrIndex(int size, int axis)
        {
            if (axis >= 0)
            {
                return axis;
            }
            else
            {
                return size + axis;
            }
        }

        public static bool CheckNoRepeatedDims(int[] dims) => dims.Length == dims.Distinct().Count();

        public static int Clamp(int value, int min, int max)
        {
            if (value < min)
            {
                return min;
            }
            else if (value > max) 
            {
                return max;
            }
            else return value;
        }

        public static int Clamp(int value, int pmin, int pmax, int nmin, int nmax)
        {
            var max = value >= 0 ? pmax : nmax;
            var min = value >= 0 ? pmin : nmin;
            if (value < min)
            {
                return min;
            }
            else if (value > max)
            {
                return max;
            }
            else return value;
        }

        public static Tuple<int[], int[]> ComputeShapesForReduction(int[] inShape, int[] axes)
        {
            List<int> shape = new List<int>();
            var rank = inShape.Length;
            for (var dim = 0; dim < rank; dim++)
            {
                if (!axes.Contains(dim))
                {
                    shape.Add(inShape[dim]);
                }
            }
            var reducedShape = axes.Select(dim => inShape[dim]).ToArray();
            return new Tuple<int[], int[]>(shape.ToArray(), reducedShape);

        }

        public static int[] GetAxesPermutationForReduction(int[] axes, int rank)
        {
            if (AxesAreInnerMostDims(axes, rank))
            {
                return null;
            }
            List<int> result = new List<int>();
            for (var i = 0; i < rank; ++i)
            {
                if (axes.ToList().IndexOf(i) == -1)
                {
                    result.Add(i);
                }
            }
            result.AddRange(axes);
            return result.ToArray();
        }
        public static bool AxesAreInnerMostDims(int[] axes, int rank)
        {
            for (var i = 0; i < axes.Length; ++i)
            {
                if (axes[axes.Length - i - 1] != rank - 1 - i)
                {
                    return false;
                }
            }
            return true;
        }

        public static int[] GetInnerMostAxes(int n, int rank)
        {
            List<int> axes = new List<int>();
            for (var i = rank - n; i < rank; ++i)
            {
                axes.Add(i);
            }
            return axes.ToArray();
        }

        public static T[,] To2DArray<T>(this T[][] source)
        {
            T[,] result = new T[source.Length, source[0].Length];

            for (int i = 0; i < source.Length; i++)
            {
                for (int k = 0; k < source[0].Length; k++)
                {
                    result[i, k] = source[i][k];
                }
            }
            return result;
        }
    }
}
