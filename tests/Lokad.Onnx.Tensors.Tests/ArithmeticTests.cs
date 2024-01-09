namespace Lokad.Onnx.Tensors.Tests
{
    public class ArithmeticTests
    {
        [Fact]
        public void CanAdd()
        {
            var a = new DenseTensor<int>(new[] { 256, 256, 3, });
            var b = a.Clone();

            a[0, 0, 1] = 1;
            b[0, 0, 1] = 1;
            var c = Tensor<int>.Add(a, b);
            Assert.Equal(2, c[0, 0, 1]);
            Assert.Equal(0, c[0, 0, 0]);
        }
    }
}