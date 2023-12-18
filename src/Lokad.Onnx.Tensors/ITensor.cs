using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace Lokad.Onnx
{
    public interface ITensor
    {
        TensorElementType ElementType { get; }
        Type PrimitiveType {get; }
        ReadOnlySpan<int> Dimensions { get; }
        string Name { get; set;  }
        ITensor Reshape_(ReadOnlySpan<int> dimensions);
    }
}
