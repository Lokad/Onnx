﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Text;
using System.Threading.Tasks;

namespace Lokad.Onnx
{
    public static class TensorExtensions
    {
        public static object GetTensorData(this TensorProto tp)
        {
            switch ((TensorElementType)tp.DataType)
            {
                case TensorElementType.Int32:
                    Runtime.Debug("tensorproto {tpn} has embedded int32 tensor data.", tp.Name);
                    return tp.Int32Data.Count == 0 && tp.RawData.Length > 0 ? MemoryMarshal.Cast<byte, int>(tp.RawData.Span).ToArray() : tp.Int32Data.ToArray();
                case TensorElementType.Int64:
                    Runtime.Debug("tensorproto {tpn} has embedded int64 tensor data.", tp.Name);
                    return tp.Int64Data.Count == 0 && tp.RawData.Length > 0 ? MemoryMarshal.Cast<byte, long>(tp.RawData.Span).ToArray() : tp.Int64Data.ToArray();
                case TensorElementType.Float: 
                    Runtime.Debug("tensorproto {tpn} has embedded float tensor data.", tp.Name);
                    return tp.FloatData.Count == 0 && tp.RawData.Length > 0 ? MemoryMarshal.Cast<byte, float>(tp.RawData.Span).ToArray() : tp.FloatData.ToArray();
                case TensorElementType.Double: 
                    Runtime.Debug("tensorproto {tpn} has embedded double tensor data.", tp.Name);
                    return tp.DoubleData.Count == 0 && tp.RawData.Length > 0 ? MemoryMarshal.Cast<byte, double>(tp.RawData.Span).ToArray() : tp.DoubleData.ToArray();
                default: throw new NotSupportedException($"Cannot get embedded tensor data of tensor element type {tp.DataType}.");
            }
        }  
        
        public static ITensor ToTensor(this TensorProto tp)
        {
            switch ((TensorElementType) tp.DataType)
            {
                case TensorElementType.Bool: return new DenseTensor<bool>(memory: (bool[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.Int8: return new DenseTensor<sbyte>(memory: (sbyte[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.UInt8: return new DenseTensor<byte>(memory: (byte[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.Int16: return new DenseTensor<short>(memory: (short[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.UInt16: return new DenseTensor<ushort>(memory: (ushort[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.Int32: return new DenseTensor<int>(memory: (int[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.UInt32: return new DenseTensor<uint>(memory: (uint[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.Int64: return new DenseTensor<long>(memory: (long[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.UInt64: return new DenseTensor<ulong>(memory: (ulong[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.Float: return new DenseTensor<float>(memory: (float[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.Double: return new DenseTensor<double>(memory: (double[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.Float16: return new DenseTensor<Float16>(memory: (Float16[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.BFloat16: return new DenseTensor<BFloat16>(memory: (BFloat16[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.Complex64: return new DenseTensor<Complex>(memory: (Complex[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                //case TensorElementType.String: return new DenseTensor<string>((string[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                default: throw new ArgumentException($"Cannot convert tensor proto of element type {tp.DataType}.");
            }
        }

        public static ITensor ToTensor(this ValueInfoProto vp)
        {
            if (vp.Type.ValueCase != TypeProto.ValueOneofCase.TensorType)
            {
                throw new ArgumentException($"The value info {vp.Name} is not a tensor type.");
            }

            switch ((TensorElementType) vp.Type.TensorType.ElemType)
            {
                case TensorElementType.Bool: return new DenseTensor<bool>(dimensions: vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.Int8: return new DenseTensor<sbyte>(dimensions: vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.UInt8: return new DenseTensor<byte>(dimensions: vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.Int16: return new DenseTensor<short>(dimensions: vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.UInt16: return new DenseTensor<ushort>(dimensions: vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.Int32: return new DenseTensor<int>(dimensions: vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.UInt32: return new DenseTensor<uint>(dimensions: vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.Int64: return new DenseTensor<long>(dimensions: vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.UInt64: return new DenseTensor<ulong>(dimensions: vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.Float: return new DenseTensor<float>(dimensions: vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.Double: return new DenseTensor<double>(dimensions: vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.Float16: return new DenseTensor<Float16>(dimensions: vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.BFloat16: return new DenseTensor<BFloat16>(dimensions: vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.Complex64: return new DenseTensor<Complex>(dimensions: vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                //case TensorElementType.String: return new DenseTensor<string>(vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                default: throw new ArgumentException($"Cannot convert value info proto of element type {vp.Type.TensorType.ElemType}.");
            }
        }
        public static string TensorNameDesc(this ValueInfoProto vp) => $"{vp.Name}:{((TensorElementType) vp.Type.TensorType.ElemType).ToString().ToLower()}:{vp.Type.TensorType.Shape.Dim.Select(d => d.DimValue.ToString()).JoinWith("x")}";

        public static string TensorNameDesc(this TensorProto vp) => $"{vp.Name}:{((TensorElementType)vp.DataType).ToString().ToLower()}:{vp.Dims.Select(d => d.ToString()).JoinWith("x")}";
    }
}
