// Logistic approximation derived from Microsoft's ONNX Runtime (MIT license),
// revision 2e2543fbe9fae542f921d47a72d21d5a4ef0b710,
// onnxruntime/core/mlas/lib/intrinsics/avx512/silu_avx512f.cpp.
// Copyright (c) Microsoft Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

namespace Lokad.Onnx;

using System;
using System.Numerics;
using System.Runtime.CompilerServices;

public partial class CPUExecutionProvider
{
    // Keep vector arithmetic and its remainder independent of the public
    // operator's original scalar loop and register allocation.
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void SigmoidRationalVector(ReadOnlySpan<float> input, Span<float> output)
    {
        int width = Vector<float>.Count;
        int i = 0;
        for (; i <= input.Length - width; i += width)
        {
            var value = new Vector<float>(input.Slice(i, width));
            var x = Vector.Max(Vector.Min(value, new Vector<float>(18f)), new Vector<float>(-18f));
            var squared = x * x;

            var numerator = Vector.FusedMultiplyAdd(squared, new Vector<float>(4.37031012579801e-11f), new Vector<float>(1.15627324459942e-07f));
            numerator = Vector.FusedMultiplyAdd(numerator, squared, new Vector<float>(6.08574864600143e-05f));
            numerator = Vector.FusedMultiplyAdd(numerator, squared, new Vector<float>(8.51377133304701e-03f));
            numerator = Vector.FusedMultiplyAdd(numerator, squared, new Vector<float>(2.48287947061529e-01f));
            numerator *= x;

            var denominator = Vector.FusedMultiplyAdd(squared, new Vector<float>(6.10247389755681e-13f), new Vector<float>(5.76102136993427e-09f));
            denominator = Vector.FusedMultiplyAdd(denominator, squared, new Vector<float>(6.29106785017040e-06f));
            denominator = Vector.FusedMultiplyAdd(denominator, squared, new Vector<float>(1.70198817374094e-03f));
            denominator = Vector.FusedMultiplyAdd(denominator, squared, new Vector<float>(1.16817656904453e-01f));
            denominator = Vector.FusedMultiplyAdd(denominator, squared, new Vector<float>(9.93151921023180e-01f));

            var activated = numerator / denominator + new Vector<float>(0.5f);
            activated = Vector.Min(Vector.Max(activated, Vector<float>.Zero), Vector<float>.One);
            activated = Vector.ConditionalSelect(Vector.Equals(value, value), activated, value);
            activated.CopyTo(output.Slice(i, width));
        }
        for (; i < input.Length; i++) output[i] = 1f / (1f + MathF.Exp(-input[i]));
    }
}
