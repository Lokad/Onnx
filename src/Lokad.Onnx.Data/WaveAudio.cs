namespace Lokad.Onnx;

using System;
using System.Buffers.Binary;
using System.IO;

/// <summary>Managed decoding of uncompressed, little-endian RIFF/WAVE recordings.</summary>
public static class WaveAudio
{
    /// <summary>Reads one mono/stereo WAV form and returns owned mono float PCM.</summary>
    /// <remarks>The readable, seekable stream starts at its current position and stays open.
    /// Supports PCM8/16/24/32 and IEEE float32/64, including extensible subtypes, at
    /// 8000..192000 Hz. Stereo channels are averaged without clipping or normalization.
    /// Empty data is silence. Overlong recordings are rejected before sample allocation.</remarks>
    public static (float[] Samples, int SampleRate) ReadMono(Stream source, TimeSpan maximumDuration)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (!source.CanRead || !source.CanSeek)
            throw new ArgumentException("WAV input must be readable and seekable.", nameof(source));
        if (maximumDuration <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(maximumDuration));
        try { return Read(source, maximumDuration); }
        catch (EndOfStreamException ex) { throw new InvalidDataException("Truncated WAV file.", ex); }
    }

    static (float[], int) Read(Stream source, TimeSpan maximumDuration)
    {
        long origin = source.Position;
        Span<byte> header = stackalloc byte[12];
        source.ReadExactly(header);
        if (header[..4].SequenceEqual("RF64"u8) || header[..4].SequenceEqual("RIFX"u8))
            throw new NotSupportedException("Only little-endian RIFF/WAVE audio is supported.");
        Require(header[..4].SequenceEqual("RIFF"u8) && header[8..].SequenceEqual("WAVE"u8), "Expected RIFF/WAVE audio.");
        long end = checked(origin + 8 + BinaryPrimitives.ReadUInt32LittleEndian(header[4..]));
        Require(end >= source.Position && end <= source.Length, "Invalid or truncated RIFF length.");
        Format? format = null;
        long dataOffset = -1;
        uint dataLength = 0;
        Span<byte> chunk = stackalloc byte[8];
        Span<byte> formatBytes = stackalloc byte[40];
        while (source.Position < end)
        {
            Require(end - source.Position >= 8, "Truncated WAV chunk header.");
            source.ReadExactly(chunk);
            uint size = BinaryPrimitives.ReadUInt32LittleEndian(chunk[4..]);
            long next = checked(source.Position + size + (size & 1));
            Require(next <= end, "WAV chunk exceeds its RIFF container.");
            if (chunk[..4].SequenceEqual("fmt "u8))
            {
                Require(format is null, "Duplicate WAV format chunk.");
                Require(size >= 16 && size != 17, "Invalid WAV format chunk length.");
                int count = (int)Math.Min(size, (uint)formatBytes.Length);
                source.ReadExactly(formatBytes[..count]);
                format = ParseFormat(formatBytes[..count], size);
            }
            else if (chunk[..4].SequenceEqual("data"u8))
            {
                Require(dataOffset < 0, "Duplicate WAV data chunk.");
                dataOffset = source.Position;
                dataLength = size;
            }
            source.Position = next;
        }
        Require(format is not null && dataOffset >= 0, "WAV requires format and data chunks.");
        var f = format.Value;
        Require(dataLength % f.Align == 0, "WAV data contains a partial sample frame.");
        long frames = dataLength / f.Align;
        Require(frames <= int.MaxValue && (decimal)frames * TimeSpan.TicksPerSecond <= (decimal)maximumDuration.Ticks * f.Rate,
            "WAV duration exceeds the requested limit.");
        var samples = new float[(int)frames];
        var buffer = new byte[4096 * f.Align];
        source.Position = dataOffset;
        for (int offset = 0; offset < samples.Length;)
        {
            int count = Math.Min(4096, samples.Length - offset);
            source.ReadExactly(buffer.AsSpan(0, count * f.Align));
            for (int i = 0; i < count; i++)
            {
                double sum = 0;
                for (int channel = 0; channel < f.Channels; channel++)
                {
                    var bytes = buffer.AsSpan(i * f.Align + channel * (f.Bits / 8), f.Bits / 8);
                    double value = Decode(bytes, f);
                    Require(double.IsFinite(value), "WAV samples must be finite.");
                    if (channel == 0) sum = value / f.Channels;
                    else sum += value / f.Channels;
                }
                float sample = (float)sum;
                Require(float.IsFinite(sample), "WAV samples exceed the float32 range.");
                samples[offset + i] = sample;
            }
            offset += count;
        }
        source.Position = end;
        return (samples, f.Rate);
    }

    readonly record struct Format(int Tag, int Channels, int Rate, int Align, int Bits, int ValidBits);

    static Format ParseFormat(ReadOnlySpan<byte> bytes, uint size)
    {
        int tag = BinaryPrimitives.ReadUInt16LittleEndian(bytes);
        int channels = BinaryPrimitives.ReadUInt16LittleEndian(bytes[2..]);
        uint rate = BinaryPrimitives.ReadUInt32LittleEndian(bytes[4..]);
        uint bytesPerSecond = BinaryPrimitives.ReadUInt32LittleEndian(bytes[8..]);
        int align = BinaryPrimitives.ReadUInt16LittleEndian(bytes[12..]);
        int bits = BinaryPrimitives.ReadUInt16LittleEndian(bytes[14..]), valid = bits;
        if (size >= 18)
            Require(18u + BinaryPrimitives.ReadUInt16LittleEndian(bytes[16..]) <= size, "Invalid WAV format extension length.");
        if (tag == 0xfffe)
        {
            Require(bytes.Length >= 40 && BinaryPrimitives.ReadUInt16LittleEndian(bytes[16..]) >= 22,
                "Truncated extensible WAV format.");
            valid = BinaryPrimitives.ReadUInt16LittleEndian(bytes[18..]);
            var subtype = new Guid(bytes[24..40]);
            if (subtype == new Guid("00000001-0000-0010-8000-00aa00389b71")) tag = 1;
            else if (subtype == new Guid("00000003-0000-0010-8000-00aa00389b71")) tag = 3;
            else throw new NotSupportedException("Unsupported extensible WAV sample format.");
        }
        if (tag != 1 && tag != 3) throw new NotSupportedException("WAV must contain uncompressed PCM or IEEE float samples.");
        if (channels is not (1 or 2)) throw new NotSupportedException("WAV must have one or two channels.");
        if (rate < 8000 || rate > 192000) throw new NotSupportedException("WAV sample rate must be between 8000 and 192000 Hz.");
        if (tag == 1 ? bits is not (8 or 16 or 24 or 32) : bits is not (32 or 64))
            throw new NotSupportedException("Unsupported WAV sample width.");
        Require(valid > 0 && valid <= bits && (tag != 3 || valid == bits), "Invalid WAV valid-bit count.");
        Require(align == channels * (bits / 8) && bytesPerSecond == rate * align, "Invalid WAV block alignment or byte rate.");
        return new Format(tag, channels, (int)rate, align, bits, valid);
    }

    static double Decode(ReadOnlySpan<byte> bytes, Format format)
    {
        if (format.Tag == 3)
            return format.Bits == 32 ? BinaryPrimitives.ReadSingleLittleEndian(bytes) : BinaryPrimitives.ReadDoubleLittleEndian(bytes);
        int raw = format.Bits switch
        {
            8 => bytes[0],
            16 => BinaryPrimitives.ReadInt16LittleEndian(bytes),
            24 => (bytes[0] | bytes[1] << 8 | bytes[2] << 16) << 8 >> 8,
            _ => BinaryPrimitives.ReadInt32LittleEndian(bytes)
        };
        int unused = format.Bits - format.ValidBits;
        Require((raw & ((1L << unused) - 1)) == 0, "Extensible PCM has nonzero unused sample bits.");
        return format.Bits == 8 ? (raw - 128) / 128.0 : raw / (double)(1L << (format.Bits - 1));
    }

    static void Require([System.Diagnostics.CodeAnalysis.DoesNotReturnIf(false)] bool condition, string message)
    {
        if (!condition) throw new InvalidDataException(message);
    }
}
