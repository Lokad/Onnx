using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.Versioning;
using System.Text;
using System.Threading.Tasks;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Lokad.Onnx.CLI;

internal class Images
{
    /// <summary>
    /// Preprocess camera images for MNIST-based neural networks.
    /// </summary>
    /// <param name="image">Source image in a byte array.</param>
    /// <returns>Preprocessed image in a byte array.</returns>
    public static byte[] Preprocess(byte[] input)
    {
        var image = Image<Rgba32>.Load<Rgba32>(input);

        var ima = Preprocess(image);

        var stream = new MemoryStream();
        image.SaveAsPng(stream);

        return stream.ToArray();
    }

    /// <summary>
    /// Preprocess camera images for MNIST-based neural networks.
    /// </summary>
    /// <param name="image">Source image in a file format agnostic structure in memory as a series of Rgba32 pixels.</param>
    /// <returns>Preprocessed image in a file format agnostic structure in memory as a series of Rgba32 pixels.</returns>
    public static Image<Rgba32> Preprocess(Image<Rgba32> image)
    {
        // Step 1: Apply a grayscale filter 
        image.Mutate(i => i.Grayscale());

        // Step 6: Downscale to 20x20
        image.Mutate(i => i.Resize(28, 28));


        return image;
    }
    public static int[] ConvertImageToArray(Image<Rgba32> image)
    {
        var pixels = new int[784];
        var i = 0;
        for (int j = 0; j < image.Height; j++)
        {
            for (int k = 0; k < image.Width; k++)
            {
                pixels[i] = 255 - ((image[k, j].R + image[k, j].G + image[k, j].B) / 3);
                i++;
            }
        }

        return pixels;
    }
}
internal class Data
{
    [RequiresPreviewFeatures]
    /*
    private static Tensor<float> PreprocessTestImage(string path)
    {
        var img = new Bitmap(path);
        var result = new float[img.Width][];

        for (int i = 0; i < img.Width; i++)
        {
            result[i] = new float[img.Height];
            for (int j = 0; j < img.Height; j++)
            {
                var pixel = img.GetPixel(i, j);

                var gray = RgbToGray(pixel);

                // Normalize the Gray value to 0-1 range
                var normalized = gray / 255;

                result[i][j] = normalized;
            }
        }
        return result;
    }
    */
    private static float RgbToGray(System.Drawing.Color pixel) => 0.299f * pixel.R + 0.587f * pixel.G + 0.114f * pixel.B;
}

