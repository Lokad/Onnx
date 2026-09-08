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
            (TensorElementType.Float16, new Float16[] { Float16.One, Float16.Zero }),
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
            new Float16[] { Float16.One, Float16.Zero, Float16.One, Float16.Zero }));
        var shape = DenseTensor<long>.OfValues(new long[] { 4 });
        Assert.Equal(OpStatus.Success, CPUExecutionProvider.Reshape(f16, shape, false, null).Status);
        Assert.Equal(OpStatus.Success, CPUExecutionProvider.Transpose(f16, new[] { 1, 0 }, null, null).Status);
        var idx = DenseTensor<int>.OfValues(new int[] { 1, 0 });
        Assert.Equal(OpStatus.Success, CPUExecutionProvider.Gather(f16, idx, 0, null).Status);
        var second = Model.ToTensor(Dto("g", TensorElementType.Float16, new[] { 2, 2 },
            new Float16[] { Float16.Zero, Float16.One, Float16.Zero, Float16.One }));
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
        b.Int32Data.Add(new int[] { 0, 1 });
        Assert.Equal(new bool[] { false, true }, (bool[])b.GetTensorData());
        var i16 = new TensorProto { Name = "i16", DataType = (int)TensorElementType.Int16 };
        i16.RawData = ByteString.CopyFrom(BitConverter.GetBytes((short)-300).Concat(BitConverter.GetBytes((short)300)).ToArray());
        Assert.Equal(new short[] { -300, 300 }, (short[])i16.GetTensorData());
        var f16 = new TensorProto { Name = "f16", DataType = (int)TensorElementType.Float16 };
        f16.RawData = ByteString.CopyFrom(new byte[] { 0x00, 0x3C, 0x00, 0xBC });
        var bits = (Float16[])f16.GetTensorData();
        Assert.Equal(2, bits.Length);
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
    public void ZerosAndOnes_WorkForSmallFloats()
    {
        Assert.Equal(Float16.One, Tensor<Float16>.Ones(2).GetValue(1));
        Assert.Equal(Float16.Zero, Tensor<Float16>.Zeros(2).GetValue(0));
        Assert.Equal(BFloat16.One, Tensor<BFloat16>.Ones(2).GetValue(1));
        Assert.Equal(BFloat16.Zero, Tensor<BFloat16>.Zeros(2).GetValue(0));
    }
}

