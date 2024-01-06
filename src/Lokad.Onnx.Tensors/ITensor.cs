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
        
        int[] Dims { get; }

        int Rank { get; }

        ITensor Clone();

        ITensor Reshape(int[] shape);

        ITensor InsertDim(int dim);
        
        ITensor PadLeft();

        ITensor BroadcastDim(int dim, int size);

        ITensor ToBroadcastedTensor();

        ITensor ToDenseTensor();
    }
}
