using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Text;
using System.Threading.Tasks;

namespace Lokad.Onnx
{
    public class TensorUtil
    {
        public static int HandleNegativeAxis(int axis, int tensorRank)
        {
            if (axis > tensorRank) throw new ArgumentException("T");
            return axis < 0 ? axis + tensorRank : axis; 
        }

        public static long HandleNegativeAxis(long axis, int tensorRank)
        {
            if (axis > tensorRank) throw new ArgumentException("T");
            return axis < 0 ? axis + tensorRank : axis;
        }

       
    }

    [RequiresPreviewFeatures]
    public static class TensorExtensions
    {
        public static object GetTensorData(this TensorProto tp)
        {
            switch ((TensorElementType)tp.DataType)
            {
                case TensorElementType.Int32:
                    Runtime.Debug($"tensorproto {tp.Name} has embedded Int32 tensor data.");
                    return tp.Int32Data.Count == 0 && tp.RawData.Length > 0 ? MemoryMarshal.Cast<byte, int>(tp.RawData.Span).ToArray() : tp.Int32Data.ToArray();
                case TensorElementType.Int64:
                    Runtime.Debug($"tensorproto {tp.Name} has embedded Int64 tensor data.");
                    return tp.Int64Data.Count == 0 && tp.RawData.Length > 0 ? MemoryMarshal.Cast<byte, long>(tp.RawData.Span).ToArray() : tp.Int64Data.ToArray();
                case TensorElementType.Float: 
                    Runtime.Debug($"tensorproto {tp.Name} has embedded Float tensor data.");
                    return tp.FloatData.Count == 0 && tp.RawData.Length > 0 ? MemoryMarshal.Cast<byte, float>(tp.RawData.Span).ToArray() : tp.FloatData.ToArray();
                case TensorElementType.Double: 
                    Runtime.Debug($"tensorproto {tp.Name} has embedded Double tensor data.");
                    return tp.DoubleData.Count == 0 && tp.RawData.Length > 0 ? MemoryMarshal.Cast<byte, double>(tp.RawData.Span).ToArray() : tp.DoubleData.ToArray();
                default: throw new NotSupportedException($"Cannot get embedded tensor data of tensor element type {tp.DataType}.");
            }
        }  
        
        public static ITensor ToTensor(this TensorProto tp)
        {
            switch ((TensorElementType) tp.DataType)
            {
                case TensorElementType.Bool: return new DenseTensor<bool>((bool[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.Int8: return new DenseTensor<sbyte>((sbyte[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.UInt8: return new DenseTensor<byte>((byte[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.Int16: return new DenseTensor<short>((short[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.UInt16: return new DenseTensor<ushort>((ushort[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.Int32: return new DenseTensor<int>((int[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.UInt32: return new DenseTensor<uint>((uint[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.Int64: return new DenseTensor<long>((long[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.UInt64: return new DenseTensor<ulong>((ulong[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.Float: return new DenseTensor<float>((float[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.Double: return new DenseTensor<double>((double[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.Float16: return new DenseTensor<Float16>((Float16[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.BFloat16: return new DenseTensor<BFloat16>((BFloat16[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
                case TensorElementType.Complex64: return new DenseTensor<Complex>((Complex[]) tp.GetTensorData(), tp.Dims.Select(d => Convert.ToInt32(d)).ToArray()) { Name = tp.Name };
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
                case TensorElementType.Bool: return new DenseTensor<bool>(vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.Int8: return new DenseTensor<sbyte>(vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.UInt8: return new DenseTensor<byte>(vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.Int16: return new DenseTensor<short>(vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.UInt16: return new DenseTensor<ushort>(vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.Int32: return new DenseTensor<int>(vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.UInt32: return new DenseTensor<uint>(vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.Int64: return new DenseTensor<long>(vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.UInt64: return new DenseTensor<ulong>(vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.Float: return new DenseTensor<float>(vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.Double: return new DenseTensor<double>(vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.Float16: return new DenseTensor<Float16>(vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.BFloat16: return new DenseTensor<BFloat16>(vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                case TensorElementType.Complex64: return new DenseTensor<Complex>(vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                //case TensorElementType.String: return new DenseTensor<string>(vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                default: throw new ArgumentException($"Cannot convert value info proto of element type {vp.Type.TensorType.ElemType}.");
            }
        }
        public static string TensorNameDesc(this ValueInfoProto vp) => $"{vp.Name}:{(TensorElementType) vp.Type.TensorType.ElemType}:{vp.Type.TensorType.Shape.Dim.Select(d => d.DimValue.ToString()).JoinWith("x")}";
    }
}
