using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace Lokad.Onnx
{
    public enum Padding { Same, Const, }; // TODO Reflect
    public enum Distribution { Uniform, Normal };
    public enum Activation { Linear, ReLU, Sigmoid, Tanh };
    public class Dimension
    {
        public int size, stride, index, padLeft, padRight;
        public Padding padType;
        public double padValue;
        public Dimension Clone() => (Dimension)MemberwiseClone();
    }

    public interface ITensor
    {
        TensorElementType ElementType { get; }
        Type PrimitiveType {get; }
        ReadOnlySpan<int> Dimensions { get; }
        ITensor Reshape_(ReadOnlySpan<int> dimensions);
    }
}
