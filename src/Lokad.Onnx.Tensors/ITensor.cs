using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;


namespace Lokad.Onnx
{
    public interface ITensor
    {
        string Name { get; set; }

        TensorElementType ElementType { get; }
        
        Type PrimitiveType {get; }
        
        ReadOnlySpan<int> Dimensions { get; }

        ITensor Reshape(int[] shape);

        ITensor Broadcast(int dim, int size);
    }
}
