namespace Lokad.Onnx.Backend.Tests;

public class ParadeeQuantizationTests
{
    [Fact]
    public void ConversionOverwritesDirtyPooledStorageAndRetainsOutstandingResults()
    {
        var input = new DenseTensor<sbyte>(Enumerable.Range(0,134).Select(i=>(sbyte)(i-128)).ToArray(),new[]{2,67});
        var scales = DenseTensor<float>.OfValues(new[]{.5f,2f});
        var zeros = DenseTensor<sbyte>.OfValues(new sbyte[]{127,-128});
        var expected = Output<float>(CPUExecutionProvider.DequantizeLinear(input,scales,zeros,0,null)).ToArray();
        var pool = new TensorBufferPool();
        var dirty = pool.Rent<float>(134);Array.Fill(dirty,float.NaN);pool.Return(dirty);
        var first = Assert.IsType<DenseTensor<float>>(Output<float>(CPUExecutionProvider.DequantizeLinear(input,scales,zeros,0,null,pool)));
        Assert.True(System.Runtime.InteropServices.MemoryMarshal.TryGetArray((ReadOnlyMemory<float>)first.Buffer,out var firstArray));
        Assert.Same(dirty,firstArray.Array);
        Assert.Equal(expected,first.ToArray());
        input.SetValue(0,0);scales.SetValue(0,1f);
        var second = Assert.IsType<DenseTensor<float>>(Output<float>(CPUExecutionProvider.DequantizeLinear(input,scales,zeros,0,null,pool)));
        Assert.True(System.Runtime.InteropServices.MemoryMarshal.TryGetArray((ReadOnlyMemory<float>)second.Buffer,out var secondArray));
        Assert.NotSame(firstArray.Array,secondArray.Array);
        Assert.Equal(-127f,second.GetValue(0));
        Assert.Equal(expected,first.ToArray());
        Assert.Equal(1,pool.Reused);
    }

    [Fact]
    public void GraphDequantizationKeepsReturnedOutputsAcrossResetAndChangedInputs()
    {
        var graph = new ComputationalGraph{Opset=new(){[""]=17}};
        graph.Metadata["Name"] = "dequantized-output-ownership";
        graph.Inputs["x"] = new DenseTensor<sbyte>(new[]{2,67});
        graph.Initializers["s"] = DenseTensor<float>.OfValues(new[]{.5f});
        graph.Outputs["y"] = new DenseTensor<float>(new[]{2,67});
        graph.Nodes.Add(new Node{Name="dq",Op=OpType.DequantizeLinear,OpTypeName="DequantizeLinear",Inputs=new[]{"x","s"},Outputs=new[]{"y"},Attributes=new(),OpsetVersion=17});
        graph.Prepare();
        var held = new List<(Tensor<float> Tensor,float[] Values)>();
        for(int repeat=0;repeat<4;repeat++)
        {
            var input = new DenseTensor<sbyte>(Enumerable.Range(0,134).Select(i=>(sbyte)(i-128+repeat)).ToArray(),new[]{2,67});
            graph.Reset();
            Assert.True(graph.Execute(new Dictionary<string,ITensor>{{"x",input}},false),graph.LastErrorMessage);
            var actual = Assert.IsAssignableFrom<Tensor<float>>(graph.Outputs["y"]);
            Assert.Equal(input.ToArray().Select(v=>v*.5f),actual.ToArray());
            held.Add((actual,actual.ToArray()));
            foreach(var previous in held) Assert.Equal(previous.Values,previous.Tensor.ToArray());
        }
    }

    static Tensor<T> Output<T>(OpResult result) where T : unmanaged
    {
        Assert.True(result.Status == OpStatus.Success, result.Message);
        return Assert.IsAssignableFrom<Tensor<T>>(Assert.Single(result.Outputs));
    }

    [Fact]
    public void HalfStorageWideningPreservesSubnormalsSignedZeroAndSpecialValues()
    {
        var values = new[] { BitConverter.UInt16BitsToHalf(0), BitConverter.UInt16BitsToHalf(0x8000),
            BitConverter.UInt16BitsToHalf(1), (Half)1.5f, Half.MaxValue, Half.PositiveInfinity, Half.NaN };
        var source = DenseTensor<Half>.OfValues(values);
        var actual = Output<float>(CPUExecutionProvider.Cast(source, TensorElementType.Float, null)).ToArray();
        Assert.Equal(new[] { 0f, -0f, 5.960464477539063e-8f, 1.5f, 65504f, float.PositiveInfinity }, actual.Take(6));
        Assert.Equal(unchecked((int)0x80000000), BitConverter.SingleToInt32Bits(actual[1]));
        Assert.True(float.IsNaN(actual[6]));
        var clone = Output<Half>(CPUExecutionProvider.Cast(source, TensorElementType.Float16, null));
        source.SetValue(3, (Half)7f);
        Assert.Equal((Half)1.5f, clone.GetValue(3));
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Cast(source, TensorElementType.Int32, null).Status);
    }

    [Fact]
    public void HalfWideningReadsLogicalStridedValues()
    {
        var source = new DenseTensor<Half>(new Half[] { (Half)1, (Half)2, (Half)3, (Half)4, (Half)5, (Half)6 }, new[] { 2, 3 });
        var view = Output<Half>(CPUExecutionProvider.Transpose(source, new[] { 1, 0 }, null, null));
        var widened = Output<double>(CPUExecutionProvider.Cast(view, TensorElementType.Double, null));
        Assert.Equal(new[] { 3, 2 }, widened.Dimensions.ToArray());
        Assert.Equal(new double[] { 1, 4, 2, 5, 3, 6 }, widened.ToArray());
    }

    [Fact]
    public void PerAxisDequantizationSubtractsZeroPointsWithoutByteOverflow()
    {
        var input = new DenseTensor<sbyte>(new sbyte[] { -128, -1, 127, 2, -20, 30 }, new[] { 2, 3 });
        var scale = DenseTensor<float>.OfValues(new[] { .5f, 2f });
        var zeros = DenseTensor<sbyte>.OfValues(new sbyte[] { -5, 10 });
        var result = Output<float>(CPUExecutionProvider.DequantizeLinear(input, scale, zeros, 0, null));
        Assert.Equal(new[] { 2, 3 }, result.Dimensions.ToArray());
        Assert.Equal(new[] { -61.5f, 2f, 66f, -16f, -60f, 40f }, result.ToArray());
        var unsigned = DenseTensor<byte>.OfValues(new byte[] { 0, 128, 255 });
        var scalar = new DenseTensor<float>(new[] { .5f }, Array.Empty<int>());
        var point = new DenseTensor<byte>(new byte[] { 128 }, Array.Empty<int>());
        Assert.Equal(new[] { -64f, 0f, 63.5f }, Output<float>(CPUExecutionProvider.DequantizeLinear(unsigned, scalar, point, null, null)).ToArray());
    }

    [Fact]
    public void DequantizationSupportsNegativeAxisAndRejectsInvalidParameters()
    {
        var input = new DenseTensor<sbyte>(new sbyte[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        var scales = DenseTensor<float>.OfValues(new[] { 1f, 2f, 3f });
        Assert.Equal(new[] { 1f, 4f, 9f, 4f, 10f, 18f }, Output<float>(CPUExecutionProvider.DequantizeLinear(input, scales, null, -1, null)).ToArray());
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.DequantizeLinear(input, scales, null, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.DequantizeLinear(input, DenseTensor<float>.OfValues(new[] { 0f }), null, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.DequantizeLinear(input, scales, DenseTensor<sbyte>.OfValues(new sbyte[] { 0 }), 1, null).Status);
        var empty = new DenseTensor<sbyte>(new[] { 2, 0 });
        Assert.Empty(Output<float>(CPUExecutionProvider.DequantizeLinear(empty, DenseTensor<float>.OfValues(new[] { 1f, 2f }), null, 0, null)).ToArray());
    }

    public static IEnumerable<object[]> ConversionLayouts()
    {
        foreach (bool unsigned in new[] { false, true })
            foreach (bool view in new[] { false, true })
                foreach (int axis in new[] { 0, 1, 2 })
                    yield return new object[] { unsigned, view, axis };
    }

    [Theory]
    [MemberData(nameof(ConversionLayouts))]
    public void ConversionBlocksMatchIntegerOracleWithTailsViewsAndLiveChanges(bool unsigned, bool view, int axis)
    {
        int[] shape = view ? new[] { 2, 67, 3 } : new[] { 2, 3, 67 };
        ITensor input = unsigned
            ? new DenseTensor<byte>(Enumerable.Range(0, 402).Select(i => (byte)(i % 256)).ToArray(), shape)
            : new DenseTensor<sbyte>(Enumerable.Range(0, 402).Select(i => (sbyte)(i % 256 - 128)).ToArray(), shape);
        if (view) input = Assert.Single(CPUExecutionProvider.Transpose(input, new[] { 0, 2, 1 }, null, null).Outputs);
        int channels = input.Dims[axis];
        var scales = DenseTensor<float>.OfValues(Enumerable.Range(0, channels).Select(i => .01337f * (i + 1)).ToArray());
        ITensor zeros = unsigned
            ? DenseTensor<byte>.OfValues(Enumerable.Range(0, channels).Select(i => (byte)(i % 2 == 0 ? 255 : 0)).ToArray())
            : DenseTensor<sbyte>.OfValues(Enumerable.Range(0, channels).Select(i => (sbyte)(i % 2 == 0 ? 127 : -128)).ToArray());
        float[] Oracle()
        {
            int[] source = unsigned ? ((Tensor<byte>)input).ToArray().Select(v => (int)v).ToArray()
                : ((Tensor<sbyte>)input).ToArray().Select(v => (int)v).ToArray();
            int[] points = unsigned ? ((Tensor<byte>)zeros).ToArray().Select(v => (int)v).ToArray()
                : ((Tensor<sbyte>)zeros).ToArray().Select(v => (int)v).ToArray();
            int inner = input.Dims.Skip(axis + 1).Aggregate(1, (a, b) => a * b);
            return source.Select((value, i) => (value - points[i / inner % channels]) * scales.GetValue(i / inner % channels)).ToArray();
        }
        var first = Output<float>(CPUExecutionProvider.DequantizeLinear(input, scales, zeros, axis, ExecutionOptions.Simd));
        var held = first.ToArray();
        for (int revision = 0; revision < 2; revision++)
        {
            var expected = Oracle().Select(BitConverter.SingleToInt32Bits).ToArray();
            foreach (var options in new[] { ExecutionOptions.Scalar, ExecutionOptions.Simd })
            {
                var actual = Output<float>(CPUExecutionProvider.DequantizeLinear(input, scales, zeros, axis == 2 ? -1 : axis, options));
                Assert.Equal(expected, actual.ToArray().Select(BitConverter.SingleToInt32Bits));
            }
            if (input is Tensor<byte> bytes) bytes.SetValue(0, 99); else ((Tensor<sbyte>)input).SetValue(0, -99);
            if (zeros is Tensor<byte> byteZeros) byteZeros.SetValue(0, 0); else ((Tensor<sbyte>)zeros).SetValue(0, -128);
            scales.SetValue(0, scales.GetValue(0) * 2);
            Assert.Equal(held, first.ToArray());
        }
    }

    [Theory]
    [InlineData(0, new long[] { 2, -1, 1, -2 })]
    [InlineData(1, new long[] { -1, -1, 1, 1 })]
    public void ModBroadcastsWithFloorOrTruncatingRemainders(int fmod, long[] expected)
    {
        var input = new DenseTensor<long>(new long[] { -7, 7 }, new[] { 2, 1 });
        var divisor = new DenseTensor<long>(new long[] { 3, -3 }, new[] { 1, 2 });
        Assert.Equal(expected, Output<long>(CPUExecutionProvider.Mod(input, divisor, fmod, null)).ToArray());
        Assert.Equal(new long[] { 0 }, Output<long>(CPUExecutionProvider.Mod(DenseTensor<long>.OfValues(new[] { long.MinValue }), DenseTensor<long>.OfValues(new long[] { -1 }), fmod, null)).ToArray());
    }

    [Fact]
    public void ModValidatesZeroAndFloatingPointRemainderFlag()
    {
        var input = DenseTensor<float>.OfValues(new[] { -7.5f, 7.5f });
        var divisor = DenseTensor<float>.OfValues(new[] { 3f });
        Assert.Equal(new[] { -1.5f, 1.5f }, Output<float>(CPUExecutionProvider.Mod(input, divisor, 1, null)).ToArray());
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Mod(input, divisor, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Mod(DenseTensor<int>.OfValues(new[] { 1 }), DenseTensor<int>.OfValues(new[] { 0 }), null, null).Status);
    }
}
