namespace Lokad.Onnx.Backend.Tests;

using System.Buffers.Binary;
using System.Text;

public class WaveAudioTests
{
    internal static byte[] Format(int tag, int channels, int rate, int bits, int validBits)
    {
        using var memory = new MemoryStream();
        using var writer = new BinaryWriter(memory);
        writer.Write((ushort)tag); writer.Write((ushort)channels); writer.Write(rate);
        writer.Write(rate * channels * bits / 8); writer.Write((ushort)(channels * bits / 8)); writer.Write((ushort)bits);
        if (tag == 0xfffe)
        {
            writer.Write((ushort)22); writer.Write((ushort)validBits); writer.Write(channels == 1 ? 4 : 3);
            writer.Write(new Guid("00000001-0000-0010-8000-00aa00389b71").ToByteArray());
        }
        return memory.ToArray();
    }

    internal static byte[] Wave(params (string Name, byte[] Data)[] chunks)
    {
        using var memory = new MemoryStream();
        using var writer = new BinaryWriter(memory);
        writer.Write("RIFF"u8); writer.Write(0); writer.Write("WAVE"u8);
        foreach (var chunk in chunks)
        {
            writer.Write(Encoding.ASCII.GetBytes(chunk.Name)); writer.Write(chunk.Data.Length); writer.Write(chunk.Data);
            if (chunk.Data.Length % 2 != 0) writer.Write((byte)0);
        }
        memory.Position = 4; writer.Write((uint)memory.Length - 8);
        return memory.ToArray();
    }

    static float[] Read(byte[] data)
    {
        using var stream = new MemoryStream(data);
        var result = WaveAudio.ReadMono(stream, TimeSpan.FromSeconds(30));
        Assert.True(stream.CanRead);
        Assert.Equal(stream.Length, stream.Position);
        Assert.Equal(16000, result.SampleRate);
        return result.Samples;
    }

    static byte[] Integers(int bits, params int[] samples)
    {
        var result = new byte[samples.Length * bits / 8];
        for (int i = 0; i < samples.Length; i++)
            for (int b = 0; b < bits / 8; b++) result[i * bits / 8 + b] = (byte)(samples[i] >> (8 * b));
        return result;
    }

    [Theory]
    [InlineData(8, 0, 128, 255, 0.9921875f)]
    [InlineData(16, -32768, 0, 32767, 0.999969482421875f)]
    [InlineData(24, -8388608, 0, 8388607, 0.9999998807907104f)]
    [InlineData(32, int.MinValue, 0, int.MaxValue, 1f)]
    public void PcmExtremaAndZeroDecode(int bits, int negative, int zero, int positive, float maximum)
    {
        Assert.Equal(new[] { -1f, 0f, maximum }, Read(Wave(("fmt ", Format(1, 1, 16000, bits, bits)),
            ("data", Integers(bits, negative, zero, positive)))));
    }

    [Theory]
    [InlineData(32)]
    [InlineData(64)]
    public void FloatSamplesPreserveVolumeAndMonoSignedZero(int bits)
    {
        using var memory = new MemoryStream();
        using (var writer = new BinaryWriter(memory, Encoding.UTF8, true))
            foreach (float value in new[] { -0f, -1.5f, 1.25f, float.Epsilon })
                if (bits == 32) writer.Write(value); else writer.Write((double)value);
        var result = Read(Wave(("fmt ", Format(3, 1, 16000, bits, bits)), ("data", memory.ToArray())));
        Assert.Equal(new[] { -0f, -1.5f, 1.25f, float.Epsilon }.Select(BitConverter.SingleToInt32Bits), result.Select(BitConverter.SingleToInt32Bits));
    }

    [Fact]
    public void StereoUsesEqualMeanWithoutIntegerOverflow()
    {
        var result = Read(Wave(("fmt ", Format(1, 2, 16000, 16, 16)),
            ("data", Integers(16, -32768, 32767, 16384, 16384, 32767, 32767))));
        Assert.Equal(new[] { -1f / 65536, 0.5f, 32767f / 32768 }, result);
    }

    [Fact]
    public void StereoLargeFloatsAverageBeforeRounding()
    {
        var result = Read(Wave(("fmt ", Format(3, 2, 16000, 32, 32)),
            ("data", BitConverter.GetBytes(float.MaxValue).Concat(BitConverter.GetBytes(float.MaxValue)).ToArray())));
        Assert.Equal(float.MaxValue, Assert.Single(result));
    }

    [Theory]
    [InlineData(24, 20, 4194304)]
    [InlineData(32, 24, 1073741824)]
    public void ExtensiblePcmValidBitsAreLeftAligned(int bits, int valid, int half)
    {
        Assert.Equal(new[] { -0.5f, 0.5f }, Read(Wave(("fmt ", Format(0xfffe, 1, 16000, bits, valid)),
            ("data", Integers(bits, -half, half)))));
    }

    [Fact]
    public void ExtensibleFloatUsesItsSubtype()
    {
        var format = Format(0xfffe, 1, 16000, 32, 32);
        new Guid("00000003-0000-0010-8000-00aa00389b71").ToByteArray().CopyTo(format, 24);
        Assert.Equal(0.25f, Assert.Single(Read(Wave(("fmt ", format), ("data", BitConverter.GetBytes(0.25f))))));
    }

    [Fact]
    public void OddUnknownChunksAndDataBeforeFormatAreSupported()
    {
        Assert.Equal(0.5f, Assert.Single(Read(Wave(("JUNK", new byte[] { 7 }), ("data", Integers(16, 16384)),
            ("fmt ", Format(1, 1, 16000, 16, 16)), ("LIST", new byte[] { 1, 2, 3 })))));
    }

    [Fact]
    public void ReadsOneFormFromCurrentPositionAndLeavesStreamOpen()
    {
        var wave = Wave(("fmt ", Format(1, 1, 16000, 16, 16)), ("data", Integers(16, 1)));
        using var stream = new MemoryStream(new byte[5].Concat(wave).Concat(new byte[7]).ToArray());
        stream.Position = 5;
        Assert.Single(WaveAudio.ReadMono(stream, TimeSpan.FromSeconds(1)).Samples);
        Assert.Equal(5 + wave.Length, stream.Position);
        Assert.True(stream.CanRead);
    }

    [Fact]
    public void ShortUnderlyingReadsAreCompleted()
    {
        using var stream = new ShortReads(Wave(("fmt ", Format(1, 1, 16000, 16, 16)), ("data", Integers(16, 16384))));
        Assert.Equal(0.5f, Assert.Single(WaveAudio.ReadMono(stream, TimeSpan.FromSeconds(1)).Samples));
    }

    sealed class ShortReads(byte[] data) : MemoryStream(data)
    {
        public override int Read(Span<byte> buffer) => base.Read(buffer[..Math.Min(3, buffer.Length)]);
    }

    [Fact]
    public void EmptyDataIsValidSilence() => Assert.Empty(Read(Wave(("fmt ", Format(1, 1, 16000, 16, 16)), ("data", Array.Empty<byte>()))));

    [Fact]
    public void DurationBoundaryIsExactAndDoesNotTruncate()
    {
        var wave = Wave(("fmt ", Format(1, 1, 16000, 16, 16)), ("data", Integers(16, 1, 2)));
        using var allowed = new MemoryStream(wave);
        Assert.Equal(2, WaveAudio.ReadMono(allowed, TimeSpan.FromTicks(1250)).Samples.Length);
        using var overlong = new MemoryStream(wave);
        Assert.Throws<InvalidDataException>(() => WaveAudio.ReadMono(overlong, TimeSpan.FromTicks(1249)));
    }

    [Theory]
    [InlineData("riff-size")]
    [InlineData("chunk-size")]
    [InlineData("short-header")]
    [InlineData("short-payload")]
    [InlineData("partial-frame")]
    [InlineData("byte-rate")]
    [InlineData("block-align")]
    [InlineData("format-size")]
    [InlineData("duplicate-format")]
    [InlineData("duplicate-data")]
    [InlineData("missing-format")]
    [InlineData("missing-data")]
    [InlineData("wrong-wave")]
    [InlineData("missing-pad")]
    public void MalformedFilesAreRefused(string kind)
    {
        var fmt = ("fmt ", Format(1, 1, 16000, 16, 16));
        var data = ("data", Integers(16, 7));
        var wave = Wave(fmt, data);
        switch (kind)
        {
            case "riff-size": BinaryPrimitives.WriteUInt32LittleEndian(wave.AsSpan(4), uint.MaxValue); break;
            case "chunk-size": BinaryPrimitives.WriteUInt32LittleEndian(wave.AsSpan(40), uint.MaxValue); break;
            case "short-header": wave = wave[..9]; break;
            case "short-payload": wave = wave[..^1]; break;
            case "partial-frame": wave = Wave(fmt, ("data", new byte[] { 1 })); break;
            case "byte-rate": wave[28]++; break;
            case "block-align": wave[32]++; break;
            case "format-size": wave[16] = 17; break;
            case "duplicate-format": wave = Wave(fmt, fmt, data); break;
            case "duplicate-data": wave = Wave(fmt, data, data); break;
            case "missing-format": wave = Wave(data); break;
            case "missing-data": wave = Wave(fmt); break;
            case "wrong-wave": wave[8] = 0; break;
            case "missing-pad": wave = Wave(fmt, data, ("JUNK", new byte[] { 1 }))[..^1]; break;
        }
        Assert.Throws<InvalidDataException>(() => Read(wave));
    }

    [Theory]
    [InlineData("unused-bits")]
    [InlineData("valid-bits")]
    [InlineData("extension-size")]
    public void InvalidExtensiblePrecisionIsRefused(string kind)
    {
        var format = Format(0xfffe, 1, 16000, 24, 20);
        var payload = Integers(24, 0x400000);
        if (kind == "unused-bits") payload[0] = 1;
        if (kind == "valid-bits") format[18] = 25;
        if (kind == "extension-size") format[16] = 30;
        Assert.Throws<InvalidDataException>(() => Read(Wave(("fmt ", format), ("data", payload))));
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(double.MaxValue)]
    public void NonfiniteOrUnrepresentableFloatSamplesAreRefused(double value) =>
        Assert.Throws<InvalidDataException>(() => Read(Wave(("fmt ", Format(3, 1, 16000, 64, 64)), ("data", BitConverter.GetBytes(value)))));

    [Theory]
    [InlineData(6, 1, 16000, 16)]
    [InlineData(1, 3, 16000, 16)]
    [InlineData(1, 1, 7999, 16)]
    [InlineData(1, 1, 192001, 16)]
    [InlineData(3, 1, 16000, 16)]
    public void UnsupportedFormatsAreRefused(int tag, int channels, int rate, int bits) =>
        Assert.Throws<NotSupportedException>(() => Read(Wave(("fmt ", Format(tag, channels, rate, bits, bits)), ("data", Array.Empty<byte>()))));

    [Theory]
    [InlineData("RF64")]
    [InlineData("RIFX")]
    public void OtherContainersAreRefused(string name)
    {
        var wave = Wave(); Encoding.ASCII.GetBytes(name).CopyTo(wave, 0);
        Assert.Throws<NotSupportedException>(() => Read(wave));
    }

    [Fact]
    public void InvalidLimitsAndUnseekableStreamsAreRefused()
    {
        using var memory = new MemoryStream();
        Assert.Throws<ArgumentOutOfRangeException>(() => WaveAudio.ReadMono(memory, TimeSpan.Zero));
        using var unseekable = new System.IO.Compression.GZipStream(memory, System.IO.Compression.CompressionMode.Decompress);
        Assert.Throws<ArgumentException>(() => WaveAudio.ReadMono(unseekable, TimeSpan.FromSeconds(1)));
    }
}
