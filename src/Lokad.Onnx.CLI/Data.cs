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
    internal static ITensor[]? GetInputTensorsFromFileArgs(IEnumerable<string> args, bool saveInput)
    {
        var op = Begin("Converting {c} file arguments to tensors", args.Count());
        var tensors = new List<ITensor>();
        int index = 0;
        foreach (string arg in args) 
        {
            var a = arg.Split("::");
            var name = a[0];
            index++;
            if (ImageExtensions.Contains(Path.GetExtension(name)))
            {
                var t = GetImageTensorFromFileArg(name, a.Length > 1 ? a[1..] : Array.Empty<string>(), index, saveInput);
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
    
    internal static ITensor? GetImageTensorFromFileArg(string name, string[] props, int index, bool saveInput) 
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
            return DenseTensor<double>.OfValues(ImageToArrayD(SaveImage(image, name, index, saveInput)));
        }
        else 
        {
            if (props[0] == "mnist")
            { 
                image.Mutate(i => i.Grayscale());
                image.Mutate(i => i.Resize(28, 28));
                return DenseTensor<double>.OfValues(ImageToArrayD(SaveImage(image, name, index, saveInput)));
            }
            else if (char.IsDigit(props[0].Split(':').First()[0]))
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
                        return DenseTensor<double>.OfValues(ImageToArrayD(SaveImage(image, name, index, saveInput)));
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
            }
        }
        return pixels;
    }

    public static float[,,,] ImageToArrayF(Image<Rgba32> image)
    {
        var pixels = new float[1, 1, image.Height, image.Width];
        for (int i = 0; i < image.Height; i++)
        {
            for (int j = 0; j < image.Width; j++)
            {
                pixels[0, 0, i, j] = 255.0f - ((image[i, j].R + image[i, j].G + image[i, j].B) / 3.0f);
            }
        }
        return pixels;
    }

    public static double[,,,] ImageToArrayD(Image<Rgba32> image)
    {
        var pixels = new double[1, 1, image.Height, image.Width];
        for (int i = 0; i < image.Height; i++)
        {
            for (int j = 0; j < image.Width; j++)
            {
                pixels[0, 0, i, j] = 255.0 - ((image[i, j].R + image[i, j].G + image[i, j].B) / 3.0);
            }
        }
        return pixels;
    }
    internal static Image<Rgba32> SaveImage(Image<Rgba32> image, string oname, int index, bool save)
    {
        var n = Path.Combine(Path.GetDirectoryName(oname)!, Path.GetFileNameWithoutExtension(oname) 
            + "_" + $"{image.Height}x{image.Width}_{index}.png");
        var stream = new FileStream(n, FileMode.Create);
        Info("Saving input image to {n}...", n);
        image.SaveAsPng(stream);
        return image;
    }

    internal static string[] ImageExtensions = new string[] { ".bmp", ".png", ".jpeg", ".jpg" };
}

