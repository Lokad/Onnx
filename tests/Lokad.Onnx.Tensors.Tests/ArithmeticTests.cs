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
            var c = a.Add(b);
            Assert.Equal(2, c[0, 0, 1]);
        }
    }
}