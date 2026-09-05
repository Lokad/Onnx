using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
namespace Lokad.Onnx
{
    public class Images : Runtime
    {
        public static ITensor? GetImageTensorFromFileArg(string name, string[] props, int index, bool saveInput)
        {
            if (!File.Exists(name))
            {
                return null;
            }
            var image = Image.Load<Rgba32>(name);
            if (image is null)
            {
                Error("Could not load file {f} as image.", name);
                return null;
            }

            Info("File {f} is {H}x{W}x{p}bpp image.", name, image.Height, image.Width, image.PixelType.BitsPerPixel);
            var n = Path.Combine(Path.GetDirectoryName(name)!, Path.GetFileNameWithoutExtension(name)
                + "_" + $"{image.Height}x{image.Width}_{index}.png");
            if (props.Length == 0)
            {
                return DenseTensor<float>.OfValues(ImageToArrayF(SaveImage(image, n, saveInput))).WithName(n);
            }
            else
            {
                if (props[0] == "mnist")
                {
                    Info("Converting image data to MINST format tensor data.");
                    image.Mutate(i => i.Grayscale());
                    image.Mutate(i => i.Resize(28, 28));
                    n = Path.Combine(Path.GetDirectoryName(name)!, Path.GetFileNameWithoutExtension(name)
                        + "_" + $"{image.Height}x{image.Width}_{index}.png");
                    return DenseTensor<float>.OfValues(ImageToArrayF(SaveImage(image, n, saveInput))).WithName(n);
                }
                else if (props[0] == "dinov2")
                {
                    Info("Converting image data to DINOv2 format tensor data.");
                    image.Mutate(i => i.Resize(224, 224));
                    n = Path.Combine(Path.GetDirectoryName(name)!, Path.GetFileNameWithoutExtension(name)
                        + "_" + $"{image.Height}x{image.Width}_{index}.png");
                    return DenseTensor<float>.OfValues(ImageToArrayF3(SaveImage(image, n, saveInput))).WithName(n);
                }
                else if (props[0] == "dinov3")
                {
                    Info("Converting image data to DINOv3 format tensor data.");
                    image.Mutate(i => i.Resize(new ResizeOptions { Size = new Size(224, 224), Mode = ResizeMode.Stretch, Sampler = KnownResamplers.Triangle })); // Triangle approximates the HF bilinear resample (resample 2); ImageSharp 3 removed Bilinear.
                    n = Path.Combine(Path.GetDirectoryName(name)!, Path.GetFileNameWithoutExtension(name)
                        + "_" + $"{image.Height}x{image.Width}_{index}.png");
                    return DenseTensor<float>.OfValues(ImageToArrayF3N(SaveImage(image, n, saveInput))).WithName(n);
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
                            n = Path.Combine(Path.GetDirectoryName(name)!, Path.GetFileNameWithoutExtension(name)
                                + "_" + $"{image.Height}x{image.Width}_{index}.png");
                            return DenseTensor<float>.OfValues(ImageToArrayF(SaveImage(image, n, saveInput))).WithName(n);
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

        public static int[,,,] ImageToArrayN(Image<Rgba32> image)
        {
            var pixels = new int[1, 1, image.Height, image.Width];
            for (int i = 0; i < image.Width; i++)
            {
                for (int j = 0; j < image.Height; j++)
                {
                    pixels[0, 0, j, i] = 255 - ((image[i, j].R + image[i, j].G + image[i, j].B) / 3);
                }
            }
            return pixels;
        }

        public static float[,,,] ImageToArrayF(Image<Rgba32> image)
        {
            var pixels = new float[1, 1, image.Height, image.Width];
            for (int i = 0; i < image.Width; i++)
            {
                for (int j = 0; j < image.Height; j++)
                {
                    pixels[0, 0, j, i] = ((image[i, j].R + image[i, j].G + image[i, j].B) / 3.0f) / 255.0f;
                }
            }
            return pixels;
        }

        public static float[,,,] ImageToArrayF3(Image<Rgba32> image)
        {
            var pixels = new float[1, 3, image.Height, image.Width];
            for (int i = 0; i < image.Width; i++)
            {
                for (int j = 0; j < image.Height; j++)
                {
                    var p = image[i, j];
                    pixels[0, 0, j, i] = p.R / 255.0f;
                    pixels[0, 1, j, i] = p.G / 255.0f;
                    pixels[0, 2, j, i] = p.B / 255.0f;
                }
            }
            return pixels;
        }        /// <summary>
        /// Converts an image to a channels-first float tensor matching the
        /// DINOv3 image processor: 0-1 rescale then ImageNet
        /// mean (0.485, 0.456, 0.406) / std (0.229, 0.224, 0.225) normalization.
        /// </summary>
        public static float[,,,] ImageToArrayF3N(Image<Rgba32> image)
        {
            var pixels = new float[1, 3, image.Height, image.Width];
            for (int i = 0; i < image.Width; i++)
            {
                for (int j = 0; j < image.Height; j++)
                {
                    var p = image[i, j];
                    pixels[0, 0, j, i] = (p.R / 255.0f - 0.485f) / 0.229f;
                    pixels[0, 1, j, i] = (p.G / 255.0f - 0.456f) / 0.224f;
                    pixels[0, 2, j, i] = (p.B / 255.0f - 0.406f) / 0.225f;
                }
            }
            return pixels;
        }

        /*
        public static float[][] ImageToArrayF(Image<Rgba32> image)
        {
            var pixels = new float[image.Height][image.Width];
            for (int i = 0; i < image.Width; i++)
            {
                for (int j = 0; j < image.Height; j++)
                {
                    pixels[j][i] = ((image[i, j].R + image[i, j].G + image[i, j].B) / 3.0f) / 255.0f;
                }
            }
            return pixels;
        }
        */
        public static double[,,,] ImageToArrayD(Image<Rgba32> image)
        {
            var pixels = new double[1, 1, image.Height, image.Width];
            for (int i = 0; i < image.Height; i++)
            {
                for (int j = 0; j < image.Width; j++)
                {
                    pixels[0, 0, j, i] = ((image[i, j].R + image[i, j].G + image[i, j].B) / 3.0) / 255.0;
                }
            }
            return pixels;
        }
        public static Image<Rgba32> SaveImage(Image<Rgba32> image, string name, bool save)
        {
            if (save)
            {
                var stream = new FileStream(name, FileMode.Create);
                if (File.Exists(name))
                {
                    Warn("Overwriting file {f} with input image.", name);
                }
                else
                {
                    Info("Saving input image to {n}.", name);
                }
                image.SaveAsPng(stream);
            }
            return image;
        }

        public static float[,,,]? LoadImageFromFile(string file)
        {
            if (!File.Exists(file))
            {
                return null;
            }
            var image = Image.Load<Rgba32>(file);
            if (image is null)
            {
                Error("Could not load file {f} as image.", file);
                return null;
            }

            Info("File {f} is {H}x{W}x{p}bpp image.", file, image.Height, image.Width, image.PixelType.BitsPerPixel);
            return ImageToArrayF(image);
        }

        public static float[,,,]? LoadMnistImageFromFile(string file)
        {
            if (!File.Exists(file))
            {
                return null;
            }
            var image = Image.Load<Rgba32>(file);
            if (image is null)
            {
                Error("Could not load file {f} as image.", file);
                return null;
            }

            Info("File {f} is {H}x{W}x{p}bpp image.", file, image.Height, image.Width, image.PixelType.BitsPerPixel);
            image.Mutate(i => i.Grayscale());
            image.Mutate(i => i.Resize(28, 28));
            return ImageToArrayF(image);
        }

        public static string[] ImageExtensions = new string[] { ".bmp", ".png", ".jpeg", ".jpg" };
    }
}
