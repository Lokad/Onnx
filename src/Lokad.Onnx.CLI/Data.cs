using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.Versioning;
using System.Text;
using System.Threading.Tasks;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Lokad.Onnx.CLI;

[RequiresPreviewFeatures]
internal class Data : Runtime
{
    internal static ITensor[]? GetInputTensorsFromFileArgs(IEnumerable<string> args)
    {
        var op = Begin("Converting {c} file arguments to tensors", args.Count());
        var tensors = new List<ITensor>();  
        foreach (string arg in args) 
        {
            var a = arg.Split("::");
            var name = a[0];
            if (ImageExtensions.Contains(Path.GetExtension(name)))
            {
                var t = GetImageTensorFromFileArg(name, a.Length > 1 ? a[1..] : Array.Empty<string>());
                if (t is not null)
                {
                    tensors.Add(t);
                }
                else
                {
                    Error("Could not convert file argument {arg} to image tensor.", name);
                    op.Abandon();
                    return null;    
                }
            }
        }
        op.Complete();
        return tensors.ToArray();
    }
    
    internal static ITensor? GetImageTensorFromFileArg(string name, string[] props) 
    {
        Program.ExitIfFileNotFound(name);
        var image = Image.Load<Rgba32>(name);
        if (image is null)
        {
            Error("Could not load file {f} as image.", name);
            return null;
        }
        
        Info("File {f} is {H}x{W}x{p}bpp", name, image.Height, image.Width, image.PixelType.BitsPerPixel);
        if (props.Length == 0)
        {
            return DenseTensor<int>.OfValues(ImageToArray(image));
        }
        else 
        {
            if (props[0] == "mnist")
            { 
                image.Mutate(i => i.Grayscale());
                image.Mutate(i => i.Resize(28, 28));
                return DenseTensor<int>.OfValues(ImageToArray(image));
            }
            else if (Char.IsDigit(props[0].Split(':').First()[0]))
            {
                if (props[0].Split(':').All(d => Int32.TryParse(d, out var _)))
                {
                    var dims = props[0].Split(':').Select(d => Int32.Parse(d)).ToArray();
                    if (dims.Length != 2)
                    {
                        Error("Cannot parse specified image dimensions {d}.", props[0]);
                        return null;
                    }
                    else
                    {
                        image.Mutate(i => i.Resize(dims[0], dims[1]));
                        return DenseTensor<int>.OfValues(ImageToArray(image));
                    }
                }
                else
                {
                    Error("Cannot parse specified image dimensions {d}.", props[0]);
                    return null;
                }
            }
            else
            {
                Error("Cannot parse specified image format {d}.", props[0]);
                return null;
            }
        }
        
    }

    public static int[,,,] ImageToArray(Image<Rgba32> image)
    {
        var pixels = new int[1, 1, image.Height, image.Width];
        for (int i = 0; i < image.Height; i++)
        {
            for (int j = 0; j < image.Width; j++)
            {
                pixels[0, 0, i, j] = 255 - ((image[i, j].R + image[i, j].G + image[i, j].B) / 3);
                i++;
            }
        }
        return pixels;
    }
  
    internal static string[] ImageExtensions = new string[] { ".bmp", ".png", ".jpeg", ".jpg" };
}

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
    
}

