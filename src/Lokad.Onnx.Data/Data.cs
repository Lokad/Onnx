﻿namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;



public class Data : Runtime
{
    public static ITensor[]? GetInputTensorsFromFileArgs(IEnumerable<string> args, bool saveInput = false)
    {
        var tensors = new List<ITensor>();
        int index = 0;
        foreach (string arg in args) 
        {
            var a = arg.Split("::");
            var name = a[0];
            index++;
            if (Images.ImageExtensions.Contains(Path.GetExtension(name)))
            {
                var t = Images.GetImageTensorFromFileArg(name, a.Length > 1 ? a[1..] : Array.Empty<string>(), index, saveInput);
                if (t is not null)
                {
                    tensors.Add(t);
                }
                else
                {
                    return null;    
                }
            }
            else if (Text.TextExtensions.Contains(Path.GetExtension(name))) 
            {
                var t = Text.GetTextTensorsFromFileArg(name, a.Skip(1).ToArray());
                if (t is not null)
                {
                    tensors.AddRange(t);    
                }
            }
        }
        return tensors.ToArray();
    }
    

}

