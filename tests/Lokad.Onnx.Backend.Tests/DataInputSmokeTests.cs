namespace Lokad.Onnx.Backend.Tests
{
    using Lokad.Onnx.Tests.Support;

    public class DataInputSmokeTests
    {
        [Fact]
        public void CanGetMnistInputTensor()
        {
            var r = Data.GetInputTensorsFromFileArgs(new[] { TestSupport.CommittedImage("mnist4.png") + "::mnist" });
            Assert.NotNull(r);
            Assert.Single(r!);
            Assert.Equal(4, r![0].Rank);
        }        [Fact]
        public void CanGetDinoV3InputTensor()
        {
            var r = Data.GetInputTensorsFromFileArgs(new[] { TestSupport.CommittedImage("mnist4.png") + "::dinov3" });
            Assert.NotNull(r);
            Assert.Single(r!);
            Assert.Equal(new[] { 1, 3, 224, 224 }, r![0].Dims.ToArray());
        }

        [Fact]
        public void DinoV3NormalizationMatchesImageNetFormula()
        {
            using var image = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32>(2, 2);
            image[0, 0] = new SixLabors.ImageSharp.PixelFormats.Rgba32(255, 0, 0);
            image[1, 0] = new SixLabors.ImageSharp.PixelFormats.Rgba32(0, 255, 0);
            image[0, 1] = new SixLabors.ImageSharp.PixelFormats.Rgba32(0, 0, 255);
            image[1, 1] = new SixLabors.ImageSharp.PixelFormats.Rgba32(255, 255, 255);
            var t = Images.ImageToArrayF3N(image);
            Assert.Equal(new[] { 1, 3, 2, 2 }, new[] { t.GetLength(0), t.GetLength(1), t.GetLength(2), t.GetLength(3) });
            Assert.Equal((1f - 0.485f) / 0.229f, t[0, 0, 0, 0], 5);
            Assert.Equal((0f - 0.456f) / 0.224f, t[0, 1, 0, 0], 5);
            Assert.Equal((0f - 0.406f) / 0.225f, t[0, 2, 0, 0], 5);
            Assert.Equal((0f - 0.485f) / 0.229f, t[0, 0, 0, 1], 5);
            Assert.Equal((1f - 0.456f) / 0.224f, t[0, 1, 0, 1], 5);
            Assert.Equal((1f - 0.406f) / 0.225f, t[0, 2, 1, 1], 5);
        }
    }
}
