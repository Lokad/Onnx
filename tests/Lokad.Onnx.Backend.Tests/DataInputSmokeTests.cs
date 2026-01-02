namespace Lokad.Onnx.Backend.Tests
{
    public class DataInputSmokeTests
    {
        [Fact]
        public void CanGetMnistInputTensor()
        {
            var r = Data.GetInputTensorsFromFileArgs(new[] { "images\\mnist4.png::mnist" });
            Assert.NotNull(r);
            Assert.Single(r!);
            Assert.Equal(4, r![0].Rank);
        }
    }
}
