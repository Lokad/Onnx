using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Lokad.Onnx.Backend
{
    public class CPUExecutionProvider
    {
        
        public static OpResult Squeeze<T>(Tensor<T> input, Tensor<int>? axes=null)
        {
            if (axes is not null && axes.Dimensions.Length != 1)
            {
                throw new ArgumentException("The axes tensor must have dimension 1.");
            }
            //ReadOnlySpan<int> _axes = axes is null ? axes.
            List<int> axesCorrected = new List<int>(input.Dimensions.Length);
            for(int i = 0; i < input.Dimensions.Length; i++) 
            {
                axesCorrected[i] = input.Dimensions[i] < 0 ? input.Dimensions[i] + input.Dimensions.Length : input.Dimensions[i];
            }
            return new OpResult(OpType.Squeeze, OpStatus.Success);
        }

        public static OpResult Squeeze(ITensor input, ITensor? axes = null)
        {
            Tensor<int>? _axes = axes is not null ? (Tensor<int>) axes : null;
            switch(input.ElementType)
            {
                case TensorElementType.Bool: return Squeeze((Tensor<bool>)input, _axes);
                case TensorElementType.Int8: return Squeeze((Tensor<byte>)input, _axes);
                case TensorElementType.UInt8: return Squeeze((Tensor<sbyte>)input, _axes);
                case TensorElementType.Int16: return Squeeze((Tensor<short>)input, _axes);
                case TensorElementType.UInt16: return Squeeze((Tensor<ushort>)input, _axes);
                case TensorElementType.Int32: return Squeeze((Tensor<int>)input, _axes);
                case TensorElementType.UInt32: return Squeeze((Tensor<uint>)input, _axes);
                case TensorElementType.Int64: return Squeeze((Tensor<long>)input, _axes);
                case TensorElementType.UInt64: return Squeeze((Tensor<ulong>)input, _axes);
                case TensorElementType.Float: return Squeeze((Tensor<float>)input, _axes);
                case TensorElementType.Double: return Squeeze((Tensor<double>)input, _axes);
                case TensorElementType.Float16: return Squeeze((Tensor<Float16>)input, _axes);
                case TensorElementType.BFloat16: return Squeeze((Tensor<BFloat16>)input, _axes);
                case TensorElementType.Complex64: return Squeeze((Tensor<System.Numerics.Complex>)input, _axes);
                default: return OpResult.NotSupported(OpType.Squeeze);
            }
        }
    }
}
