using Lokad.Onnx.Tests.Support;

namespace Lokad.Onnx.Backend.Tests;

public class NpySupportTests
{
    static T WithFile<T>(byte[] bytes, Func<string, T> fn)
    {
        string path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".npy");
        try
        {
            File.WriteAllBytes(path, bytes);
            return fn(path);
        }
        finally { File.Delete(path); }
    }

    static void ExpectInvalid(byte[] bytes)
    {
        WithFile(bytes, p => { Assert.Throws<InvalidDataException>(() => NpySupport.ReadTensor(p)); return 0; });
    }

    static byte[] PayloadOf<T>(T[] values) where T : unmanaged
    {
        var raw = new byte[values.Length * System.Runtime.InteropServices.Marshal.SizeOf<T>()];
        Buffer.BlockCopy(values, 0, raw, 0, raw.Length);
        return raw;
    }

    static byte[] Build(string descr, int[] shape, byte[] payload, int major = 1, bool fortran = false, bool dropFortran = false, byte[] extra = null)
    {
        string shapeText = shape.Length == 0 ? "()" : "(" + string.Join(", ", shape) + (shape.Length == 1 ? "," : string.Empty) + ")";
        string orderPart = dropFortran ? string.Empty : "'fortran_order': " + (fortran ? "True" : "False") + ", ";
        string dict = "{'descr': " + (char)34 + descr + (char)34 + "', " + orderPart + "'shape': " + shapeText + ", }";
        int pre = major == 1 ? 10 : 12;
        string header = dict;
        while ((pre + header.Length + 1) % 64 != 0) header += " ";
        header += "\n";
        byte[] hb = System.Text.Encoding.ASCII.GetBytes(header);
        byte[] lb = major == 1
            ? new byte[] { (byte)(hb.Length & 0xFF), (byte)(hb.Length >> 8) }
            : new byte[] { (byte)(hb.Length & 0xFF), (byte)((hb.Length >> 8) & 0xFF), (byte)((hb.Length >> 16) & 0xFF), (byte)((hb.Length >> 24) & 0xFF) };
        var file = new System.Collections.Generic.List<byte>();
        file.AddRange(new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', (byte)major, (byte)0 });
        file.AddRange(lb);
        file.AddRange(hb);
        file.AddRange(payload);
        if (extra != null) file.AddRange(extra);
        return file.ToArray();
    }

    [Fact]
    public void Float32_RoundTrips()
    {
        byte[] bytes = Build("<f4", new[] { 3 }, PayloadOf(new float[] { 1f, 2f, 3f }));
        var t = WithFile(bytes, p => NpySupport.ReadTensor(p));
        var f = Assert.IsAssignableFrom<Tensor<float>>(t);
        Assert.Equal(new[] { 3 }, f.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 2f, 3f }, f.ToArray());
    }

    [Fact]
    public void Int32_Scalar_And_Int64_Matrix_RoundTrip()
    {
        byte[] s32 = Build("<i4", Array.Empty<int>(), PayloadOf(new int[] { 42 }));
        var t32 = WithFile(s32, p => NpySupport.ReadTensor(p));
        var f32 = Assert.IsAssignableFrom<Tensor<int>>(t32);
        Assert.Empty(f32.Dimensions.ToArray());
        Assert.Equal(new int[] { 42 }, f32.ToArray());
        byte[] s64 = Build("<i8", new[] { 2, 1 }, PayloadOf(new long[] { 7L, -8L }));
        var t64 = WithFile(s64, p => NpySupport.ReadTensor(p));
        var f64 = Assert.IsAssignableFrom<Tensor<long>>(t64);
        Assert.Equal(new[] { 2, 1 }, f64.Dimensions.ToArray());
        Assert.Equal(new long[] { 7L, -8L }, f64.ToArray());
    }

    [Fact]
    public void Version2_Header_Accepted()
    {
        byte[] bytes = Build("<f4", new[] { 2 }, PayloadOf(new float[] { 0.5f, -1f }), 2);
        var (values, shape) = WithFile(bytes, p => NpySupport.ReadFloat32(p));
        Assert.Equal(new[] { 2 }, shape);
        Assert.Equal(new float[] { 0.5f, -1f }, values);
    }

    [Fact]
    public void BadMagic_Throws()
    {
        ExpectInvalid(new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    }

    [Fact]
    public void FortranOrder_Throws()
    {
        ExpectInvalid(Build("<f4", new[] { 1 }, PayloadOf(new float[] { 1f }), 1, true));
    }

    [Fact]
    public void UnsupportedDtype_Throws()
    {
        ExpectInvalid(Build("<f8", new[] { 1 }, new byte[8]));
    }

    [Fact]
    public void BigEndianDtype_Throws()
    {
        ExpectInvalid(Build(">f4", new[] { 1 }, PayloadOf(new float[] { 1f })));
    }

    [Fact]
    public void TruncatedPayload_Throws()
    {
        byte[] full = Build("<f4", new[] { 3 }, PayloadOf(new float[] { 1f, 2f, 3f }));
        ExpectInvalid(full[0..^1]);
    }

    [Fact]
    public void TrailingByte_Throws()
    {
        byte[] bytes = Build("<f4", new[] { 1 }, PayloadOf(new float[] { 1f }), 1, false, false, new byte[] { 0 });
        ExpectInvalid(bytes);
    }

    [Fact]
    public void ShapeProductMismatch_Throws()
    {
        ExpectInvalid(Build("<f4", new[] { 4 }, PayloadOf(new float[] { 1f, 2f, 3f })));
    }

    [Fact]
    public void MissingFortranOrder_Throws()
    {
        ExpectInvalid(Build("<f4", new[] { 1 }, PayloadOf(new float[] { 1f }), 1, false, true));
    }

    [Fact]
    public void ReadFloat32_WrongDtype_Throws()
    {
        byte[] bytes = Build("<i4", new[] { 1 }, PayloadOf(new int[] { 5 }));
        WithFile(bytes, p => { Assert.Throws<InvalidDataException>(() => NpySupport.ReadFloat32(p)); return 0; });
    }
}
