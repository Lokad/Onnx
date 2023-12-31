namespace Lokad.Onnx.Backend.Tests
{
    public class OpTests
    {
        [Fact]
        public void SqueezeTests()
        {

            var d = new DenseTensor<int>(new[] { 1, 2, 2, 3, 4, 4, 5, 4 });
            var sd = d.Dimensions.ToArray();
            Assert.Equal( new[] { 1, 2, 3, 4, 5, 4 }, sd);
        }

        [Fact]
        public void BroadcastTests() 
        {
            var a = new DenseTensor<int>(new[] { 256, 256, 3, });
            var b = new DenseTensor<int>(new[] { 3 });

            var r = CPUExecutionProvider.Broadcast(a, b);
            Assert.Equal(OpStatus.Success, r.Status);
        }
    }
}
