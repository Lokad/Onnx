using Lokad.Onnx.Tests.Support;

namespace Lokad.Onnx.Backend.Tests;

public class NpySupportTests
{
    // Base test set adapted from voice 2b1138f; boundary cases below were added
    // during selective import. No model download is needed for parser tests.
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

    static byte[] Build(string descr, int[] shape, byte[] payload) =>
        Build(descr, shape, payload, 1);

    static byte[] Build(string descr, int[] shape, byte[] payload, int major) =>
        Build(descr, shape, payload, major, false);

    static byte[] Build(string descr, int[] shape, byte[] payload, int major, bool fortran) =>
        Build(descr, shape, payload, major, fortran, false);

    static byte[] Build(string descr, int[] shape, byte[] payload, int major, bool fortran, bool dropFortran) =>
        Build(descr, shape, payload, major, fortran, dropFortran, Array.Empty<byte>());

    static byte[] Build(string descr, int[] shape, byte[] payload, int major, bool fortran, bool dropFortran, byte[] extra)
    {
        string shapeText = shape.Length == 0 ? "()" : "(" + string.Join(", ", shape) + (shape.Length == 1 ? "," : string.Empty) + ")";
        string orderPart = dropFortran ? string.Empty : "'fortran_order': " + (fortran ? "True" : "False") + ", ";
        string dict = "{'descr': '" + descr + "', " + orderPart + "'shape': " + shapeText + ", }";
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
        file.AddRange(extra);
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

    static byte[] WithHeader(string header)
    {
        var bytes = new List<byte> { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 1, 0 };
        var encoded = System.Text.Encoding.ASCII.GetBytes(header + "\n");
        bytes.Add((byte)encoded.Length);
        bytes.Add((byte)(encoded.Length >> 8));
        bytes.AddRange(encoded);
        return bytes.ToArray();
    }

    [Fact]
    public void NumpyGeneratedInt64FixturePreservesValuesBeyondDoublePrecision()
    {
        // NumPy 2.2.4: np.save(BytesIO(), np.array([[2**60+3,-2**60-7]], dtype=np.int64)).
        byte[] bytes = Convert.FromBase64String("k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDIpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoDAAAAAAAAEPn////////v");
        var tensor = Assert.IsAssignableFrom<Tensor<long>>(WithFile(bytes, NpySupport.ReadTensor));
        Assert.Equal(new[] { 1, 2 }, tensor.Dimensions.ToArray());
        Assert.Equal(new[] { (1L << 60) + 3, -(1L << 60) - 7 }, tensor.ToArray());
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void EverySupportedVersionPreservesFloatBits(int version)
    {
        var values = new[] { -0.0f, float.PositiveInfinity, float.NegativeInfinity, BitConverter.Int32BitsToSingle(0x7fc12345) };
        var read = WithFile(Build("<f4", new[] { 2, 2 }, PayloadOf(values), version), NpySupport.ReadFloat32);
        Assert.Equal(new[] { 2, 2 }, read.Shape);
        Assert.Equal(values.Select(BitConverter.SingleToInt32Bits), read.Values.Select(BitConverter.SingleToInt32Bits));
    }

    [Fact]
    public void EmptyTensorAcceptsReorderedDoubleQuotedFieldsAndWhitespace()
    {
        var tensor = Assert.IsAssignableFrom<Tensor<int>>(WithFile(WithHeader(
            " {\"shape\" : (2, 0, 4), \"fortran_order\" : False, \"descr\" : \"<i4\" } "), NpySupport.ReadTensor));
        Assert.Equal(new[] { 2, 0, 4 }, tensor.Dimensions.ToArray());
        Assert.Empty(tensor.ToArray());
    }

    [Theory]
    [InlineData("{'descr':'<f4','fortran_order':False,'shape':(0,), 'descr':'<i4'}")]
    [InlineData("{'descr':'<f4','fortran_order':False,'shape':(0,), 'extra':False}")]
    [InlineData("{'descr':'<f4','fortran_order':False,'shape':(0,)} garbage")]
    [InlineData("{'descr':'<f4','fortran_order':False,'shape':(0)}")]
    [InlineData("{'descr':'<f4','fortran_order':False,'shape':(0,,)}")]
    [InlineData("{'descr':'<f4','fortran_order':False,'shape':(,0)}")]
    [InlineData("{'descr':'<f4','fortran_order':False,'shape':(-1,)}")]
    [InlineData("{'descr':'<f4','fortran_order':False,'shape':(2147483648,)}")]
    [InlineData("{'descr':'<f4','fortran_order':False,'shape':(2147483647,2147483647,2147483647)}")]
    [InlineData("{'descr':'<f4','fortran_order':False,'shape':(3.0,)}")]
    [InlineData("{'descr':'<f4','fortran_order':'False','shape':(0,)}")]
    [InlineData("{'descr':False,'fortran_order':False,'shape':(0,)}")]
    [InlineData("{'descr':'<f4' 'fortran_order':False,'shape':(0,)}")]
    public void MalformedDictionaryThrows(string header) => ExpectInvalid(WithHeader(header));

    [Theory]
    [InlineData(0x7fffffff)]
    [InlineData(0x80000000)]
    [InlineData(0xffffffff)]
    public void OversizedHeaderLengthThrowsWithoutIntegerWraparound(uint length)
    {
        byte[] file = Build("<f4", new[] { 0 }, Array.Empty<byte>(), 2);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(file.AsSpan(8, 4), length);
        ExpectInvalid(file);
    }

    [Fact]
    public void TruncatedHeaderUnknownVersionAndMissingNewlineThrow()
    {
        byte[] valid = Build("<f4", new[] { 0 }, Array.Empty<byte>());
        for (int size = 0; size < valid.Length; size++) ExpectInvalid(valid[..size]);
        var badMinor = (byte[])valid.Clone();
        badMinor[7] = 1;
        ExpectInvalid(badMinor);
        var badMajor = (byte[])valid.Clone();
        badMajor[6] = 4;
        ExpectInvalid(badMajor);
        var noNewline = (byte[])valid.Clone();
        noNewline[^1] = (byte)' ';
        ExpectInvalid(noNewline);
    }
}
