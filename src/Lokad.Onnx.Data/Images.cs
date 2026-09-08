using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using static Lokad.Onnx.Runtime;
namespace Lokad.Onnx
{
    public class Images
    {
        public static ITensor? GetImageTensorFromFileArg(string name, string[] props, int index, bool saveInput)
        {
            if (!File.Exists(name))
            {
                return null;
            }
            using var image = Image.Load<Rgba32>(name);
            if (image is null)
            {
                Error("Could not load file {f} as image.", name);
                return null;
            }

            Info("File {f} is {H}x{W}x{p}bpp image.", name, image.Height, image.Width, image.PixelType.BitsPerPixel);
            var n = Path.Combine((Path.GetDirectoryName(name) ?? ""), Path.GetFileNameWithoutExtension(name)
                + "_" + $"{image.Height}x{image.Width}_{index}.png");
            if (props.Length == 0)
            {
                return ImageToTensorF(SaveImage(image, n, saveInput)).WithName(n);
            }
            else
            {
                if (props[0] == "mnist")
                {
                    Info("Converting image data to MINST format tensor data.");
                    image.Mutate(i => i.Grayscale());
                    image.Mutate(i => i.Resize(28, 28));
                    n = Path.Combine((Path.GetDirectoryName(name) ?? ""), Path.GetFileNameWithoutExtension(name)
                        + "_" + $"{image.Height}x{image.Width}_{index}.png");
                    return ImageToTensorF(SaveImage(image, n, saveInput)).WithName(n);
                }
                else if (props[0] == "dinov2")
                {
                    Info("Converting image data to DINOv2 format tensor data.");
                    image.Mutate(i => i.Resize(224, 224));
                    n = Path.Combine((Path.GetDirectoryName(name) ?? ""), Path.GetFileNameWithoutExtension(name)
                        + "_" + $"{image.Height}x{image.Width}_{index}.png");
                    return ImageToTensorF3(SaveImage(image, n, saveInput)).WithName(n);
                }
                else if (props[0] == "dinov3")
                {
                    Info("Converting image data to DINOv3 format tensor data.");
                    image.Mutate(i => i.Resize(new ResizeOptions { Size = new Size(224, 224), Mode = ResizeMode.Stretch, Sampler = KnownResamplers.Triangle })); // Triangle approximates the HF bilinear resample (resample 2); ImageSharp 3 removed Bilinear.
                    n = Path.Combine((Path.GetDirectoryName(name) ?? ""), Path.GetFileNameWithoutExtension(name)
                        + "_" + $"{image.Height}x{image.Width}_{index}.png");
                    return ImageToTensorF3N(SaveImage(image, n, saveInput)).WithName(n);
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
                            n = Path.Combine((Path.GetDirectoryName(name) ?? ""), Path.GetFileNameWithoutExtension(name)
                                + "_" + $"{image.Height}x{image.Width}_{index}.png");
                            return ImageToTensorF(SaveImage(image, n, saveInput)).WithName(n);
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

        /// <summary>
        /// Converts an image straight into a channels-first float tensor,
        /// writing row by row through pixel spans with no intermediate array.
        /// Values match ImageToArrayF exactly: row-average gray over 255.
        /// </summary>
        public static DenseTensor<float> ImageToTensorF(Image<Rgba32> image)
        {
            int w = image.Width, h = image.Height;
            var flat = new float[h * w];
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < h; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    int b = y * w;
                    for (int x = 0; x < w; x++)
                    {
                        var p = row[x];
                        flat[b + x] = ((p.R + p.G + p.B) / 3.0f) / 255.0f;
                    }
                }
            });
            return new DenseTensor<float>(new Memory<float>(flat), new[]{1, 1, h, w});
        }

        /// <summary>
        /// Converts an image straight into a 1-by-3-by-height-by-width float
        /// tensor, writing row by row through pixel spans with no intermediate
        /// array. Values match ImageToArrayF3 exactly: per-channel over 255.
        /// </summary>
        public static DenseTensor<float> ImageToTensorF3(Image<Rgba32> image)
        {
            int w = image.Width, h = image.Height;
            var flat = new float[3 * h * w];
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < h; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    int b = y * w;
                    for (int x = 0; x < w; x++)
                    {
                        var pix = row[x];
                        flat[b + x] = pix.R / 255.0f;
                        flat[h * w + b + x] = pix.G / 255.0f;
                        flat[2 * h * w + b + x] = pix.B / 255.0f;
                    }
                }
            });
            return new DenseTensor<float>(new Memory<float>(flat), new[]{1, 3, h, w});
        }

        /// <summary>
        /// Converts an image straight into a normalized channels-first float
        /// tensor matching the DINOv3 image processor: 0-1 rescale then
        /// ImageNet mean (0.485, 0.456, 0.406) / std (0.229, 0.224, 0.225)
        /// normalization. Values match ImageToArrayF3N exactly.
        /// </summary>
        public static DenseTensor<float> ImageToTensorF3N(Image<Rgba32> image)
        {
            int w = image.Width, h = image.Height;
            var flat = new float[3 * h * w];
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < h; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    int b = y * w;
                    for (int x = 0; x < w; x++)
                    {
                        var pix = row[x];
                        flat[b + x] = (pix.R / 255.0f - 0.485f) / 0.229f;
                        flat[h * w + b + x] = (pix.G / 255.0f - 0.456f) / 0.224f;
                        flat[2 * h * w + b + x] = (pix.B / 255.0f - 0.406f) / 0.225f;
                    }
                }
            });
            return new DenseTensor<float>(new Memory<float>(flat), new[]{1, 3, h, w});
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
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    pixels[0, 0, y, x] = ((image[x, y].R + image[x, y].G + image[x, y].B) / 3.0) / 255.0;
                }
            }
            return pixels;
        }
        public static Image<Rgba32> SaveImage(Image<Rgba32> image, string name, bool save)
        {
            if (save)
            {
                if (File.Exists(name))
                {
                    Warn("Overwriting file {f} with input image.", name);
                }
                else
                {
                    Info("Saving input image to {n}.", name);
                }
                using var stream = new FileStream(name, FileMode.Create);
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
            using var image = Image.Load<Rgba32>(file);
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
            using var image = Image.Load<Rgba32>(file);
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
