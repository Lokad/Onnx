extern alias OnnxSharp;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Google.Protobuf;
using OnnxSharp::Onnx;

namespace Lokad.Onnx.Backend.Tests;

public class DtypeMappingTests
{
    static OnnxTensor Dto(string name, TensorElementType type, int[] dims, Array data) =>
        new OnnxTensor { Name = name, ElementType = type, Dims = dims, Data = data };

    [Fact]
    public void RoundTrips_PreserveTypeDimsAndValues()
    {
        var cases = new (TensorElementType Type, Array Data)[]
        {
            (TensorElementType.Bool, new bool[] { true, false }),
            (TensorElementType.Int8, new sbyte[] { -1, 5 }),
            (TensorElementType.UInt8, new byte[] { 255, 5 }),
            (TensorElementType.Int16, new short[] { -300, 300 }),
            (TensorElementType.UInt16, new ushort[] { 65535, 7 }),
            (TensorElementType.Int32, new int[] { -7, 7 }),
            (TensorElementType.UInt32, new uint[] { 4000000000u, 7u }),
            (TensorElementType.Int64, new long[] { -9L, 9L }),
            (TensorElementType.UInt64, new ulong[] { 18000000000000000000ul, 9ul }),
            (TensorElementType.Float, new float[] { 1.5f, -2.5f }),
            (TensorElementType.Double, new double[] { 1.25, -2.5 }),
            (TensorElementType.Float16, new Half[] { Half.One, Half.Zero }),
            (TensorElementType.BFloat16, new BFloat16[] { BFloat16.One, BFloat16.Zero }),
            (TensorElementType.Complex64, new Complex[] { new Complex(1, 2), new Complex(3, 4) }),
        };
        foreach (var (type, data) in cases)
        {
            var tensor = Model.ToTensor(Dto("v", type, new[] { 2 }, data));
            Assert.Equal(type, tensor.ElementType);
            Assert.Equal(new[] { 2 }, tensor.Dims);
            Assert.Equal(data.GetValue(0), tensor.GetValue(0));
            Assert.Equal(data.GetValue(1), tensor.GetValue(1));
        }
    }

    [Fact]
    public void UnsupportedDescriptors_FailExplicitly()
    {
        Assert.Throws<ArgumentException>(() => Model.ToTensor(Dto("s", TensorElementType.String, new[] { 1 }, new string[] { "hi" })));
    }

    [Fact]
    public void Reshape_KeepsSignedness()
    {
        var si = Model.ToTensor(Dto("si", TensorElementType.Int8, new[] { 2 }, new sbyte[] { -1, 2 }));
        var shape = DenseTensor<long>.OfValues(new long[] { 1, 2 });
        var rs = CPUExecutionProvider.Reshape(si, shape, false, null);
        Assert.Equal(OpStatus.Success, rs.Status);
        Assert.Equal(TensorElementType.Int8, rs.Outputs[0].ElementType);
        Assert.Equal((sbyte)-1, rs.Outputs[0].GetValue(0));
        var su = Model.ToTensor(Dto("su", TensorElementType.UInt8, new[] { 2 }, new byte[] { 255, 2 }));
        var ru = CPUExecutionProvider.Reshape(su, shape, false, null);
        Assert.Equal(OpStatus.Success, ru.Status);
        Assert.Equal(TensorElementType.UInt8, ru.Outputs[0].ElementType);
        Assert.Equal((byte)255, ru.Outputs[0].GetValue(0));
    }

    [Fact]
    public void Float16_ShapeOps_Succeed()
    {
        var f16 = Model.ToTensor(Dto("f", TensorElementType.Float16, new[] { 2, 2 },
            new Half[] { Half.One, Half.Zero, Half.One, Half.Zero }));
        var shape = DenseTensor<long>.OfValues(new long[] { 4 });
        Assert.Equal(OpStatus.Success, CPUExecutionProvider.Reshape(f16, shape, false, null).Status);
        Assert.Equal(OpStatus.Success, CPUExecutionProvider.Transpose(f16, new[] { 1, 0 }, null, null).Status);
        var idx = DenseTensor<int>.OfValues(new int[] { 1, 0 });
        Assert.Equal(OpStatus.Success, CPUExecutionProvider.Gather(f16, idx, 0, null).Status);
        var second = Model.ToTensor(Dto("g", TensorElementType.Float16, new[] { 2, 2 },
            new Half[] { Half.Zero, Half.One, Half.Zero, Half.One }));
        Assert.Equal(OpStatus.Success, CPUExecutionProvider.Concat(new ITensor[] { f16, second }, 0, null).Status);
    }

    [Fact]
    public void Constant_AcceptsLongScalarsAndArrays()
    {
        var s = CPUExecutionProvider.Constant(5L, null);
        Assert.Equal(OpStatus.Success, s.Status);
        Assert.Equal(TensorElementType.Int64, s.Outputs[0].ElementType);
        Assert.Equal(5L, s.Outputs[0].GetValue(0));
        var a = CPUExecutionProvider.Constant(new long[] { 1L, 2L }, null);
        Assert.Equal(OpStatus.Success, a.Status);
        Assert.Equal(TensorElementType.Int64, a.Outputs[0].ElementType);
        Assert.Equal(new[] { 2 }, a.Outputs[0].Dims);
    }

    [Fact]
    public void Constant_RejectsStringsExplicitly()
    {
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Constant("hi", null).Status);
    }

    [Fact]
    public void Import_ReadsBoolAndWidthsAndPackedSmallFloats()
    {
        var b = new TensorProto { Name = "b", DataType = (int)TensorElementType.Bool };
        b.Dims.Add(2);
        b.Int32Data.Add(new int[] { 0, 1 });
        Assert.Equal(new bool[] { false, true }, (bool[])b.GetTensorData());
        var i16 = new TensorProto { Name = "i16", DataType = (int)TensorElementType.Int16 };
        i16.Dims.Add(2);
        i16.RawData = ByteString.CopyFrom(BitConverter.GetBytes((short)-300).Concat(BitConverter.GetBytes((short)300)).ToArray());
        Assert.Equal(new short[] { -300, 300 }, (short[])i16.GetTensorData());
        var f16 = new TensorProto { Name = "f16", DataType = (int)TensorElementType.Float16 };
        f16.Dims.Add(2);
        f16.RawData = ByteString.CopyFrom(new byte[] { 0x00, 0x3C, 0x00, 0xBC });
        var bits = (Half[])f16.GetTensorData();
        Assert.Equal(2, bits.Length);
        Assert.Equal(new Half[] { (Half)1f, (Half)(-1f) }, bits);
    }

    [Fact]
    public void Import_RejectsStringsAndComplexExplicitly()
    {
        var s = new TensorProto { Name = "s", DataType = (int)TensorElementType.String };
        Assert.Throws<NotSupportedException>(() => s.GetTensorData());
        var c = new TensorProto { Name = "c", DataType = (int)TensorElementType.Complex64 };
        var ex = Assert.Throws<NotSupportedException>(() => c.GetTensorData());
        Assert.Contains("c", ex.Message);
    }

    [Fact]
    public void Import_RejectsComplex128Explicitly()
    {
        // Complex128 has no dispatch arm, no dense factory, and no
        // importer path: the refusal names the tensor like Complex64.
        var c = new TensorProto { Name = "c128", DataType = (int)TensorElementType.Complex128 };
        var ex = Assert.Throws<NotSupportedException>(() => c.GetTensorData());
        Assert.Contains("c128", ex.Message);
    }

    [Fact]
    public void ElementArray_RejectsComplex128()
    {
        // The dense-array factory stops at float64; complex128 has no
        // dense representation by design (complex64 works only through
        // the generic Tensor<T> path, never this factory).
        Assert.Throws<NotSupportedException>(() => TensorBase.CreateElementArray(TensorElementType.Complex128, 2));
    }

    [Fact]
    public void Import_ReadsUnsignedExtremaLosslessly()
    {
        var u64 = new TensorProto { Name = "u64", DataType = (int)TensorElementType.UInt64 };
        u64.Dims.Add(2);
        u64.Uint64Data.Add(new ulong[] { ulong.MaxValue, 0UL });
        Assert.Equal(new ulong[] { ulong.MaxValue, 0UL }, (ulong[])u64.GetTensorData());
        var u32 = new TensorProto { Name = "u32", DataType = (int)TensorElementType.UInt32 };
        u32.Dims.Add(2);
        u32.Uint64Data.Add(new ulong[] { uint.MaxValue, 0UL });
        Assert.Equal(new uint[] { uint.MaxValue, 0U }, (uint[])u32.GetTensorData());
        var raw = new TensorProto { Name = "u64raw", DataType = (int)TensorElementType.UInt64 };
        raw.Dims.Add(1);
        raw.RawData = ByteString.CopyFrom(BitConverter.GetBytes(ulong.MaxValue));
        Assert.Equal(new ulong[] { ulong.MaxValue }, (ulong[])raw.GetTensorData());
        var legacy = new TensorProto { Name = "u64leg", DataType = (int)TensorElementType.UInt64 };
        legacy.Dims.Add(2);
        legacy.Int64Data.Add(new long[] { 7L, 8L });
        Assert.Equal(new ulong[] { 7UL, 8UL }, (ulong[])legacy.GetTensorData());
    }

    [Fact]
    public void Import_RejectsMalformedPayloads()
    {
        var shortp = new TensorProto { Name = "shortp", DataType = (int)TensorElementType.Int32 };
        shortp.Dims.Add(2); shortp.Dims.Add(2);
        shortp.Int32Data.Add(new int[] { 1, 2, 3 });
        var ex = Assert.Throws<InvalidOperationException>(() => shortp.GetTensorData());
        Assert.Contains("shortp", ex.Message);
        var longp = new TensorProto { Name = "longp", DataType = (int)TensorElementType.Int32 };
        longp.Dims.Add(2); longp.Dims.Add(2);
        longp.Int32Data.Add(new int[] { 1, 2, 3, 4, 5 });
        ex = Assert.Throws<InvalidOperationException>(() => longp.GetTensorData());
        Assert.Contains("longp", ex.Message);
        var ragged = new TensorProto { Name = "ragged", DataType = (int)TensorElementType.Float };
        ragged.Dims.Add(1);
        ragged.RawData = ByteString.CopyFrom(new byte[] { 0x00, 0x00, 0x80 });
        ex = Assert.Throws<InvalidOperationException>(() => ragged.GetTensorData());
        Assert.Contains("ragged", ex.Message);
        var wide = new TensorProto { Name = "wide", DataType = (int)TensorElementType.UInt32 };
        wide.Dims.Add(1);
        wide.Uint64Data.Add(new ulong[] { (ulong)uint.MaxValue + 1UL });
        Assert.Throws<OverflowException>(() => wide.GetTensorData());
    }

    [Fact]
    public void Import_SkipsCountCheckForUnderivableDims()
    {
        var sym = new TensorProto { Name = "sym", DataType = (int)TensorElementType.Int32 };
        sym.Dims.Add(-1);
        sym.Int32Data.Add(new int[] { 1, 2 });
        Assert.Equal(new int[] { 1, 2 }, (int[])sym.GetTensorData());
    }

    [Fact]
    public void ZerosAndOnes_WorkForSmallFloats()
    {
        Assert.Equal(Half.One, Tensor<Half>.Ones(2).GetValue(1));
        Assert.Equal(Half.Zero, Tensor<Half>.Zeros(2).GetValue(0));
        Assert.Equal(BFloat16.One, Tensor<BFloat16>.Ones(2).GetValue(1));
        Assert.Equal(BFloat16.Zero, Tensor<BFloat16>.Zeros(2).GetValue(0));
    }
}

