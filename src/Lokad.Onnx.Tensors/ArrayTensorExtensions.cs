// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Versioning;

namespace Lokad.Onnx
{
    public static class ArrayTensorExtensions
    {
        /// <summary>
        /// Creates a copy of this n-dimensional array as a DenseTensor&lt;T&gt;
        /// </summary>
        /// <typeparam name="T">Type contained in the array to copy to the DenseTensor&lt;T&gt;.</typeparam>
        /// <param name="array">The array to create a DenseTensor&lt;T&gt; from.</param>
        /// <param name="reverseStride">False (default) to indicate that the first dimension is most major (farthest apart) and the last dimension is most minor (closest together): akin to row-major in a rank-2 tensor.  True to indicate that the last dimension is most major (farthest apart) and the first dimension is most minor (closest together): akin to column-major in a rank-2 tensor.</param>
        /// <returns>A n-dimensional DenseTensor&lt;T&gt; with the same dimensions and content as <paramref name="array"/>.</returns>
        public static DenseTensor<T> ToTensor<T>(this Array array, bool reverseStride = false) where T :  struct
            => new DenseTensor<T>(array, reverseStride);

        /// <summary>
        /// Creates a copy of this single-dimensional array as a SparseTensor&lt;T&gt;
        /// </summary>
        /// <typeparam name="T">Type contained in the array to copy to the SparseTensor&lt;T&gt;.</typeparam>
        /// <param name="array">The array to create a SparseTensor&lt;T&gt; from.</param>
        /// <returns>A 1-dimensional SparseTensor&lt;T&gt; with the same length and content as <paramref name="array"/>.</returns>
        public static SparseTensor<T> ToSparseTensor<T>(this T[] array) where T :  struct
        {
            return new SparseTensor<T>(array);
        }

        /// <summary>
        /// Creates a copy of this two-dimensional array as a SparseTensor&lt;T&gt;
        /// </summary>
        /// <typeparam name="T">Type contained in the array to copy to the SparseTensor&lt;T&gt;.</typeparam>
        /// <param name="array">The array to create a SparseTensor&lt;T&gt; from.</param>
        /// <param name="reverseStride">False (default) to indicate that the first dimension is most major (farthest apart) and the last dimension is most minor (closest together): row-major.  True to indicate that the last dimension is most major (farthest apart) and the first dimension is most minor (closest together): column-major.</param>
        /// <returns>A 2-dimensional SparseTensor&lt;T&gt; with the same dimensions and content as <paramref name="array"/>.</returns>
        public static SparseTensor<T> ToSparseTensor<T>(this T[,] array, bool reverseStride = false) where T :  struct
        {
            return new SparseTensor<T>(array, reverseStride);
        }

        /// <summary>
        /// Creates a copy of this three-dimensional array as a SparseTensor&lt;T&gt;
        /// </summary>
        /// <typeparam name="T">Type contained in the array to copy to the SparseTensor&lt;T&gt;.</typeparam>
        /// <param name="array">The array to create a SparseTensor&lt;T&gt; from.</param>
        /// <param name="reverseStride">False (default) to indicate that the first dimension is most major (farthest apart) and the last dimension is most minor (closest together): akin to row-major in a rank-2 tensor.  True to indicate that the last dimension is most major (farthest apart) and the first dimension is most minor (closest together): akin to column-major in a rank-2 tensor.</param>
        /// <returns>A 3-dimensional SparseTensor&lt;T&gt; with the same dimensions and content as <paramref name="array"/>.</returns>
        public static SparseTensor<T> ToSparseTensor<T>(this T[,,] array, bool reverseStride = false) where T :  struct
        {
            return new SparseTensor<T>(array, reverseStride);
        }

        /// <summary>
        /// Creates a copy of this n-dimensional array as a SparseTensor&lt;T&gt;
        /// </summary>
        /// <typeparam name="T">Type contained in the array to copy to the SparseTensor&lt;T&gt;.</typeparam>
        /// <param name="array">The array to create a SparseTensor&lt;T&gt; from.</param>
        /// <param name="reverseStride">False (default) to indicate that the first dimension is most major (farthest apart) and the last dimension is most minor (closest together): akin to row-major in a rank-2 tensor.  True to indicate that the last dimension is most major (farthest apart) and the first dimension is most minor (closest together): akin to column-major in a rank-2 tensor.</param>
        /// <returns>A n-dimensional SparseTensor&lt;T&gt; with the same dimensions and content as <paramref name="array"/>.</returns>
        public static SparseTensor<T> ToSparseTensor<T>(this Array array, bool reverseStride = false) where T :  struct
        {
            return new SparseTensor<T>(array, reverseStride);
        }

        /// <summary>
        /// Creates a copy of this single-dimensional array as a CompressedSparseTensor&lt;T&gt;
        /// </summary>
        /// <typeparam name="T">Type contained in the array to copy to the CompressedSparseTensor&lt;T&gt;.</typeparam>
        /// <param name="array">The array to create a CompressedSparseTensor&lt;T&gt; from.</param>
        /// <returns>A 1-dimensional CompressedSparseTensor&lt;T&gt; with the same length and content as <paramref name="array"/>.</returns>
        public static CompressedSparseTensor<T> ToCompressedSparseTensor<T>(this T[] array) where T :  struct
        {
            return new CompressedSparseTensor<T>(array);
        }

        /// <summary>
        /// Creates a copy of this two-dimensional array as a CompressedSparseTensor&lt;T&gt;
        /// </summary>
        /// <typeparam name="T">Type contained in the array to copy to the CompressedSparseTensor&lt;T&gt;.</typeparam>
        /// <param name="array">The array to create a CompressedSparseTensor&lt;T&gt; from.</param>
        /// <param name="reverseStride">False (default) to indicate that the first dimension is most major (farthest apart) and the last dimension is most minor (closest together): row-major.  True to indicate that the last dimension is most major (farthest apart) and the first dimension is most minor (closest together): column-major.</param>
        /// <returns>A 2-dimensional CompressedSparseTensor&lt;T&gt; with the same dimensions and content as <paramref name="array"/>.</returns>
        public static CompressedSparseTensor<T> ToCompressedSparseTensor<T>(this T[,] array, bool reverseStride = false) where T :  struct
        {
            return new CompressedSparseTensor<T>(array, reverseStride);
        }

        /// <summary>
        /// Creates a copy of this three-dimensional array as a CompressedSparseTensor&lt;T&gt;
        /// </summary>
        /// <typeparam name="T">Type contained in the array to copy to the CompressedSparseTensor&lt;T&gt;.</typeparam>
        /// <param name="array">The array to create a CompressedSparseTensor&lt;T&gt; from.</param>
        /// <param name="reverseStride">False (default) to indicate that the first dimension is most major (farthest apart) and the last dimension is most minor (closest together): akin to row-major in a rank-2 tensor.  True to indicate that the last dimension is most major (farthest apart) and the first dimension is most minor (closest together): akin to column-major in a rank-2 tensor.</param>
        /// <returns>A 3-dimensional CompressedSparseTensor&lt;T&gt; with the same dimensions and content as <paramref name="array"/>.</returns>
        public static CompressedSparseTensor<T> ToCompressedSparseTensor<T>(this T[,,] array, bool reverseStride = false) where T :  struct
        {
            return new CompressedSparseTensor<T>(array, reverseStride);
        }

        /// <summary>
        /// Creates a copy of this n-dimensional array as a CompressedSparseTensor&lt;T&gt;
        /// </summary>
        /// <typeparam name="T">Type contained in the array to copy to the CompressedSparseTensor&lt;T&gt;.</typeparam>
        /// <param name="array">The array to create a CompressedSparseTensor&lt;T&gt; from.</param>
        /// <param name="reverseStride">False (default) to indicate that the first dimension is most major (farthest apart) and the last dimension is most minor (closest together): akin to row-major in a rank-2 tensor.  True to indicate that the last dimension is most major (farthest apart) and the first dimension is most minor (closest together): akin to column-major in a rank-2 tensor.</param>
        /// <returns>A n-dimensional CompressedSparseTensor&lt;T&gt; with the same dimensions and content as <paramref name="array"/>.</returns>
        public static CompressedSparseTensor<T> ToCompressedSparseTensor<T>(this Array array, bool reverseStride = false) where T :  struct
        {
            return new CompressedSparseTensor<T>(array, reverseStride);
        }

        public static string Print<T>(this IEnumerable<T> a) => "[" + a.Select(v => a.ToString()).Aggregate((p, n) => p + ", " + n) + "]";
    }
}
