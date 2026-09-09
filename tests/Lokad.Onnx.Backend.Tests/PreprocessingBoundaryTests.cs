namespace Lokad.Onnx.Backend.Tests;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Lokad.Onnx.Tests.Support;

[Collection("ProcessState")]
public class PreprocessingBoundaryTests
{
    static string WritePng(int w, int h, Func<int, int, Rgba32> pixel, out string directory)
    {
        directory = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, "in.png");
        using var image = new Image<Rgba32>(w, h);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                image[x, y] = pixel(x, y);
        image.SaveAsPng(path);
        return path;
    }

    [Fact]
    public void Dinov3_NonSquare_StretchesTo224()
    {
        var path = WritePng(100, 50, (x, y) => x < 50 ? new Rgba32(255, 0, 0) : new Rgba32(0, 0, 255), out var directory);
        try
        {
            var t = Images.GetImageTensorFromFileArg(path, new[] { "dinov3" }, 0, false);
            Assert.NotNull(t);
            var f = (Tensor<float>)t!;
            Assert.Equal(new int[] { 1, 3, 224, 224 }, f.Dimensions.ToArray());
            Assert.True(f[0, 0, 112, 10] > 1.0f && f[0, 2, 112, 10] < 0f, "left interior must stay red");
            Assert.True(f[0, 2, 112, 213] > 1.0f && f[0, 0, 112, 213] < 0f, "right interior must stay blue");
        }
        finally
        {
            Directory.Delete(directory, true);
        }
    }

    [Fact]
    public void Dinov2_ChannelOrderWithoutNormalization()
    {
        using var image = new Image<Rgba32>(2, 1);
        image[0, 0] = new Rgba32(255, 0, 0);
        image[1, 0] = new Rgba32(0, 0, 255);
        var f = Images.ImageToTensorF3(image);
        Assert.Equal(new int[] { 1, 3, 1, 2 }, f.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 0f }, new float[] { f[0, 0, 0, 0], f[0, 0, 0, 1] });
        Assert.Equal(new float[] { 0f, 0f }, new float[] { f[0, 1, 0, 0], f[0, 1, 0, 1] });
        Assert.Equal(new float[] { 0f, 1f }, new float[] { f[0, 2, 0, 0], f[0, 2, 0, 1] });
    }

    [Fact]
    public void Dinov3_NormalizationFormula()
    {
        using var image = new Image<Rgba32>(1, 1);
        image[0, 0] = new Rgba32(255, 0, 0);
        var f = Images.ImageToTensorF3N(image);
        Assert.Equal((1f - 0.485f) / 0.229f, f[0, 0, 0, 0], 4);
        Assert.Equal((0f - 0.456f) / 0.224f, f[0, 1, 0, 0], 4);
        Assert.Equal((0f - 0.406f) / 0.225f, f[0, 2, 0, 0], 4);
    }

    [Fact]
    public void Dinov2FileArg_NonSquare_Produces224()
    {
        var path = WritePng(60, 120, (x, y) => new Rgba32(0, 255, 0), out var directory);
        try
        {
            var t = Images.GetImageTensorFromFileArg(path, new[] { "dinov2" }, 0, false);
            Assert.NotNull(t);
            Assert.Equal(new int[] { 1, 3, 224, 224 }, ((Tensor<float>)t!).Dimensions.ToArray());
        }
        finally
        {
            Directory.Delete(directory, true);
        }
    }

    static string TokenizerAsset() =>
        ModelFixture.RequireModelOrSkip("e5 tokenizer", "models", "multilingual-e5-small", "sentencepiece.bpe.model");

    [SkippableFact]
    public void LongText_TruncatesTo512Deterministically()
    {
        var text = string.Concat(System.Linq.Enumerable.Repeat("word ", 2000));
        var first = Text.RobertaTokenizeFromFile(text, TokenizerAsset());
        Assert.NotNull(first);
        var ids = (Tensor<long>)first![0];
        Assert.Equal(512, ids.Dimensions[1]);
        Assert.Equal(0, ids[0, 0]);
        var second = Text.RobertaTokenizeFromFile(text, TokenizerAsset());
        Assert.NotNull(second);
        Assert.Equal(ids.ToArray(), ((Tensor<long>)second![0]).ToArray());
        var mask = (Tensor<long>)first[1];
        Assert.Equal(512, mask.Dimensions[1]);
        for (int i = 0; i < 512; i++) Assert.Equal(1, mask[0, i]);
    }
}
