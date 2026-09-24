using System.Text.Json;
using System.Runtime.InteropServices;
using Lokad.Onnx;

static partial class Program
{
    static Prepared<T> Prepare<T>(JsonElement spec) where T : unmanaged
    {
        string name = Text(spec, "name"), dtype = Text(spec, "dtype"), layout = Text(spec, "layout"), mask = Text(spec, "mask");
        var cs = Shape(spec, "cshape"); var xs = Shape(spec, "xshape"); var ys = Shape(spec, "yshape");
        var cdata = new bool[Count(cs)]; var xdata = Values<T>(Count(xs), spec.TryGetProperty("x_seed", out var seed) ? seed.GetInt32() : 4); var ydata = Values<T>(Count(ys), 0);
        if (spec.TryGetProperty("files", out var files))
        {
            cdata = File.ReadAllBytes(Path.Combine(Base, "fixtures", files[0].GetString()!)).Select(v => v != 0).ToArray();
            xdata = MemoryMarshal.Cast<byte, T>(File.ReadAllBytes(Path.Combine(Base, "fixtures", files[1].GetString()!))).ToArray();
            ydata = MemoryMarshal.Cast<byte, T>(File.ReadAllBytes(Path.Combine(Base, "fixtures", files[2].GetString()!))).ToArray();
        }
        if (mask != "captured")
            for (int i = 0; i < cdata.Length; i++) cdata[i] = mask == "true" || mask == "first" && i == 0 || mask == "last" && i == cdata.Length - 1 || mask == "alternating" && i % 2 == 0;
        if (spec.TryGetProperty("raw_mask", out var rawMask))
            cdata = MemoryMarshal.Cast<byte, bool>(rawMask.EnumerateArray().Select(v => v.GetByte()).ToArray()).ToArray();
        var ci = Make(cdata, cs, layout, "condition"); var xi = Make(xdata, xs, layout, "x"); var yi = Make(ydata, ys, layout, "y");
        if (layout == "alias") xi = new(new DenseTensor<T>(yi.Store.AsMemory(3, 1), xs), yi.Store);
        var c = layout == "null-condition" ? null! : ci.Tensor; var x = layout == "null-x" ? null! : xi.Tensor; var y = layout == "null-y" ? null! : yi.Tensor;
        var expected = Oracle(c, x, y, out var shape);
        return new(ci, xi, yi, expected, shape);
    }
}
