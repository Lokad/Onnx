// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Collections.Generic;
using System.Globalization;
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
        public static DenseTensor<T> ToTensor<T>(this Array array) where T : unmanaged
            => array.ToTensor<T>(false);

        public static DenseTensor<T> ToTensor<T>(this Array array, bool reverseStride) where T : unmanaged
            => new DenseTensor<T>(array, reverseStride);

        public static string Print<T>(this IEnumerable<T> a) => "[" + a.Select(v => a.ToString()).Aggregate((p, n) => p + ", " + n) + "]";

        public static T[] Copy<T>(this T[] a)
        {
            var b = new T[a.Length];
            a.CopyTo(b, 0); 
            return b;
        }

        public static U[] CastA<U>(this Array a) => a.Cast<U>().ToArray();

        public static U[] Convert<T, U>(this T[] a) => 
            a.Select(e => (U?) System.Convert.ChangeType(e, typeof(U)) ?? throw new ArgumentException()).ToArray();

        public static bool AlmostEqual(this double expected, double actual, int precision)
        {
            double num = Math.Round(expected, precision);
            double num2 = Math.Round(actual, precision);
            return num == num2;
        }

        public static bool AlmostEqual(this IEnumerable<double> expected, IEnumerable<double> actual, int precision) =>
            expected.Zip(actual).All(fs => fs.First.AlmostEqual(fs.Second, precision));

        public static bool AlmostEqual(this float expected, float actual, int precision)
        {
            float num = MathF.Round(expected, precision);
            float num2 = MathF.Round(actual, precision);
            return num == num2;
        }

        public static bool AlmostEqual(this IEnumerable<float> expected, IEnumerable<float> actual, int precision) =>
            expected.Zip(actual).All(fs => fs.First.AlmostEqual(fs.Second, precision));
    }
}
