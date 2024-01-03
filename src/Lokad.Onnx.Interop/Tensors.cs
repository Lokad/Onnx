namespace Lokad.Onnx.Interop;

using System;
using Python.Runtime;

public class Tensors
{
    public static ITensor MakeTensor<T>(int[] dims) => new DenseTensor<T>(dims);
    
    public static int[] GetDims(ITensor tensor) => tensor.Dimensions.ToArray();

    //public ITensor BroadcastDim(ITensor tensor, int dim, int size) 
    //{
    //    if (tensor is DenseTensor<int> idt) { return idt.b}
        
    //};
}

