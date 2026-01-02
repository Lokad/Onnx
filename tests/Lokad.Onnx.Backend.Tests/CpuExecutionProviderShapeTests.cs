using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests
{
    public class CpuExecutionProviderShapeTests
    {
        [Fact]
        public void CanGetShape()
        {
            var t = DenseTensor<float>.OfShape(3, 4, 5);
            var o = CPU.Shape(t, 1);
            Assert.Equal(OpStatus.Success, o.Status);
            var shape = (Tensor<long>)o.Outputs![0];
            Assert.Equal(new long[] { 4, 5 }, shape.ToArray());
        }
    }
}
