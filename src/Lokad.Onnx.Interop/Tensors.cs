namespace Lokad.Onnx.Interop;

public class Tensors
{
    public static ITensor MakeTensor<T>(int[] dims) where T : struct => new DenseTensor<T>(dims);

    public static ITensor ARange(int start, int end, int step = 1) => Tensor<int>.Arange(start, end, step);
}

