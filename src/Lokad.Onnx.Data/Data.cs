namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static Lokad.Onnx.Runtime;



public class Data
{
    public static ITensor[]? GetInputTensorsFromFileArgs(IEnumerable<string> args) => GetInputTensorsFromFileArgs(args, false);

    public static ITensor[]? GetInputTensorsFromFileArgs(IEnumerable<string> args, bool saveInput)
    {
        var op = Begin("Converting {c} file argument(s) to tensors", args.Count());
        var tensors = new List<ITensor>();
        int index = 0;
        foreach (string arg in args)
        {
            // Each descriptor is parsed once into a file name plus option
            // segments; empty segments are dropped so a trailing "::" cannot
            // crash the format parsers downstream.
            var a = arg.Split("::");
            var name = a[0];
            var props = a.Skip(1).Where(s => !string.IsNullOrEmpty(s)).ToArray();
            index++;
            if (string.IsNullOrEmpty(name))
            {
                Error("Empty input file argument {arg}.", arg);
                op.Abandon();
                return null;
            }
            var extension = Path.GetExtension(name);
            if (Images.ImageExtensions.Contains(extension, StringComparer.OrdinalIgnoreCase))
            {
                var t = Images.GetImageTensorFromFileArg(name, props, index, saveInput);
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
            else if (Text.TextExtensions.Contains(extension, StringComparer.OrdinalIgnoreCase))
            {
                var t = Text.GetTextTensorsFromFileArg(name, props);
                if (t is not null)
                {
                    tensors.AddRange(t);
                }
                else
                {
                    Error("Could not convert file argument {arg} to text tensors.", name);
                    op.Abandon();
                    return null;
                }
            }
            else
            {
                Error("Unsupported input file type {ext} for argument {arg}.", extension, arg);
                op.Abandon();
                return null;
            }
        }
        op.Complete();
        return tensors.ToArray();
    }
    

}

