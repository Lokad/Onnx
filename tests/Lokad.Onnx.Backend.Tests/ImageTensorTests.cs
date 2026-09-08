
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Lokad.Onnx.Backend.Tests;

public class ImageTensorTests
{
    static Image<Rgba32> RectImage()
    {
        var image = new Image<Rgba32>(5, 3);
        for (int y = 0; y < 3; y++)
            for (int x = 0; x < 5; x++)
                image[x, y] = new Rgba32((byte)(x * 17), (byte)(y * 41), (byte)(x + y));
        return image;
    }

    static float[] Flat(float[,,,] a)
    {
        var list = new System.Collections.Generic.List<float>(
            a.GetLength(0) * a.GetLength(1) * a.GetLength(2) * a.GetLength(3));
        foreach (var v in a) list.Add(v);
        return list.ToArray();
    }

    [Fact]
    public void DirectWriters_MatchArrayHelpers_Rectangular()
    {
        using var image = RectImage();
        Assert.Equal(Flat(Images.ImageToArrayF(image)), ((Tensor<float>)Images.ImageToTensorF(image)).ToArray());
        Assert.Equal(Flat(Images.ImageToArrayF3(image)), ((Tensor<float>)Images.ImageToTensorF3(image)).ToArray());
        Assert.Equal(Flat(Images.ImageToArrayF3N(image)), ((Tensor<float>)Images.ImageToTensorF3N(image)).ToArray());
        var t = (Tensor<float>)Images.ImageToTensorF3(image);
        Assert.Equal(new int[] { 1, 3, 3, 5 }, t.Dimensions.ToArray());
    }

    [Fact]
    public void ImageToArrayD_Rectangular_Correct()
    {
        using var image = RectImage();
        var d = Images.ImageToArrayD(image);
        Assert.Equal(new int[] { 1, 1, 3, 5 }, new int[] { d.GetLength(0), d.GetLength(1), d.GetLength(2), d.GetLength(3) });
        for (int y = 0; y < 3; y++)
            for (int x = 0; x < 5; x++)
            {
                var p = image[x, y];
                double expected = ((p.R + p.G + p.B) / 3.0) / 255.0;
                Assert.Equal(expected, d[0, 0, y, x]);
            }
    }

    [Fact]
    public void SaveReload_RoundTrips_AndReopensImmediately()
    {
        var dir = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        Directory.CreateDirectory(dir);
        try
        {
            var saved = Path.Combine(dir, "r.png");
            using (var image = RectImage())
            {
                Images.SaveImage(image, saved, true);
            }
            Assert.True(File.Exists(saved));
            var bytes = File.ReadAllBytes(saved);
            Assert.True(bytes.Length > 0);
            using (var reloaded = Image.Load<Rgba32>(saved))
            {
                Assert.Equal(5, reloaded.Width);
                Assert.Equal(3, reloaded.Height);
            }
            using (var image = RectImage())
            {
                Images.SaveImage(image, saved, true);
            }
            Assert.True(File.ReadAllBytes(saved).Length > 0);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void RepeatedLoadSave_Stable()
    {
        var dir = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        Directory.CreateDirectory(dir);
        try
        {
            var copy = Path.Combine(dir, "m.png");
            File.Copy(Path.Combine(AppContext.BaseDirectory, "images", "mnist4.png"), copy);
            float[]? first = null;
            for (int i = 0; i < 3; i++)
            {
                var t = Images.GetImageTensorFromFileArg(copy, new[] { "mnist" }, i, true);
                Assert.NotNull(t);
                var flat = ((Tensor<float>)t!).ToArray();
                if (first is null) first = flat;
                else Assert.Equal(first, flat);
            }
        }
        finally { Directory.Delete(dir, true); }
    }
}
