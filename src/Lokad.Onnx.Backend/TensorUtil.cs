using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
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

    public static class TensorExtensions
    {
        public static ITensor ToTensor(this ValueInfoProto vp)
        {
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
                case TensorElementType.String: return new DenseTensor<string>(vp.Type.TensorType.Shape.Dim.Select(d => Convert.ToInt32(d.DimValue)).ToArray()) { Name = vp.Name };
                default: throw new ArgumentException($"Cannot convert tensor element type {vp.Type.TensorType.ElemType}.");
            }
        }
    }
}
