using Lokad.Onnx;
using System.Buffers.Binary;
using System.Globalization;
using System.Text.RegularExpressions;

namespace Lokad.Onnx.Tests.Support;

/// <summary>
/// Test and benchmark fixture reader for NumPy .npy files (ships nowhere).
/// Single consolidation of the previously duplicated header/payload parsers:
/// validates magic, version, header bounds, little-endian dtype, C order,
/// shape product, and exact payload size before materializing tensors.
/// Supported dtypes are float32 ("<f4"), int32 ("<i4"), int64 ("<i8").
/// </summary>
internal static class NpySupport
{
    // Adapted from voice branch 2b1138f (d80ed3d, ce25752). Fixture tooling only.
    internal static ITensor ReadTensor(string path)
    {
        var (kind, shape, payload) = ReadCore(path);
        if (kind == "<f4") return ToTensor<float>(payload, shape);
        if (kind == "<i4") return ToTensor<int>(payload, shape);
        return ToTensor<long>(payload, shape);
    }

    internal static (float[] Values, int[] Shape) ReadFloat32(string path)
    {
        var (kind, shape, payload) = ReadCore(path);
        if (kind != "<f4")
            throw new InvalidDataException("Expected float32 NPY: " + path + " (got " + kind + ").");
        var values = new float[payload.Length / 4];
        Buffer.BlockCopy(payload, 0, values, 0, payload.Length);
        return (values, shape);
    }

    static (string Kind, int[] Shape, byte[] Payload) ReadCore(string path)
    {
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("NPY fixture decoding requires a little-endian host.");
        byte[] raw = File.ReadAllBytes(path);
        if (raw.Length < 10)
            throw new InvalidDataException("Truncated NPY file: " + path + ".");
        byte[] magic = new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' };
        for (int i = 0; i < magic.Length; i++)
            if (raw[i] != magic[i])
                throw new InvalidDataException("Bad NPY magic: " + path + ".");
        int major = raw[6];
        if (raw[7] != 0)
            throw new InvalidDataException("Unsupported NPY minor version: " + path + ".");
        int headerOff;
        int headerLen;
        if (major == 1)
        {
            headerOff = 10;
            headerLen = raw[8] | (raw[9] << 8);
        }
        else if (major == 2 || major == 3)
        {
            if (raw.Length < 12)
                throw new InvalidDataException("Truncated NPY header: " + path + ".");
            headerOff = 12;
            uint length = BinaryPrimitives.ReadUInt32LittleEndian(raw.AsSpan(8, 4));
            if (length > int.MaxValue)
                throw new InvalidDataException("NPY header overruns file: " + path + ".");
            headerLen = (int)length;
        }
        else throw new InvalidDataException("Unsupported NPY version " + major + ": " + path + ".");
        if (headerLen == 0 || headerLen > raw.Length - headerOff)
            throw new InvalidDataException("NPY header overruns file: " + path + ".");
        if (raw[headerOff + headerLen - 1] != (byte)'\n')
            throw new InvalidDataException("NPY header must end in a newline: " + path + ".");
        // Only simple numeric dtypes are supported, so their header grammar is
        // ASCII even for format 3. Reject lossy decoding of malformed bytes.
        for (int i = headerOff; i < headerOff + headerLen; i++)
            if (raw[i] > 127) throw new InvalidDataException("Non-ASCII numeric NPY header: " + path + ".");
        string header = System.Text.Encoding.ASCII.GetString(raw, headerOff, headerLen);
        var fields = ParseFields(header, path);
        string kind = fields["descr"];
        if (kind != "<f4" && kind != "<i4" && kind != "<i8")
            throw new InvalidDataException("Unsupported NPY dtype " + kind + ": " + path + ".");
        if (fields["fortran_order"] != "False")
            throw new InvalidDataException("Fortran-order NPY is not supported: " + path + ".");
        int[] shape = ParseShape(fields["shape"], path);
        long n = 1;
        try
        {
            foreach (int d in shape) n = checked(n * d);
            n = checked(n * (kind == "<i8" ? 8 : 4));
        }
        catch (OverflowException ex)
        {
            throw new InvalidDataException("NPY shape product is too large: " + path + ".", ex);
        }
        long expected = n;
        long actual = raw.Length - headerOff - headerLen;
        if (actual != expected)
            throw new InvalidDataException("NPY payload size mismatch: " + path + " (expected " + expected + " payload bytes, file holds " + actual + ").");
        var payload = new byte[expected];
        Array.Copy(raw, headerOff + headerLen, payload, 0, payload.Length);
        return (kind, shape, payload);
    }

    // Parse only the numeric NPY dictionary grammar; never evaluate Python.
    // Key order, quote style and whitespace are flexible. Duplicates, extra
    // fields, malformed tuples and text after the dictionary are rejected.
    static readonly Regex Field = new(
        "\\G\\s*(?<q>['\"])(?<key>descr|fortran_order|shape)\\k<q>\\s*:\\s*" +
        "(?:(?<vquote>['\"])(?<dtype>[^'\"\\r\\n]*)\\k<vquote>|(?<order>True|False)|(?<shape>\\([^()]*\\)))\\s*",
        RegexOptions.CultureInvariant, TimeSpan.FromSeconds(1));

    static Dictionary<string, string> ParseFields(string header, string path)
    {
        string text = header.Trim();
        if (!text.StartsWith('{') || !text.EndsWith('}'))
            throw new InvalidDataException("Malformed NPY dictionary: " + path + ".");
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        int position = 1;
        while (position < text.Length - 1)
        {
            var match = Field.Match(text, position);
            if (!match.Success) throw new InvalidDataException("Malformed NPY field: " + path + ".");
            string key = match.Groups["key"].Value;
            string valueGroup = key == "descr" ? "dtype" : key == "shape" ? "shape" : "order";
            if (!match.Groups[valueGroup].Success || !result.TryAdd(key, match.Groups[valueGroup].Value))
                throw new InvalidDataException("Duplicate or invalid NPY field: " + path + ".");
            position = match.Index + match.Length;
            if (position == text.Length - 1) break;
            if (text[position] != ',') throw new InvalidDataException("Malformed NPY dictionary separator: " + path + ".");
            position++;
            while (position < text.Length - 1 && char.IsWhiteSpace(text[position])) position++;
        }
        if (result.Count != 3) throw new InvalidDataException("NPY header requires descr, fortran_order and shape: " + path + ".");
        return result;
    }

    static int[] ParseShape(string tuple, string path)
    {
        string inner = tuple[1..^1].Trim();
        if (inner.Length == 0) return Array.Empty<int>();
        string[] parts = inner.Split(',');
        var dimList = new System.Collections.Generic.List<int>();
        for (int i = 0; i < parts.Length; i++)
        {
            string part = parts[i].Trim();
            if (part.Length == 0 && i == parts.Length - 1) continue;
            if (!int.TryParse(part, NumberStyles.None, CultureInfo.InvariantCulture, out int value))
                throw new InvalidDataException("NPY header has invalid dimension: " + path + ".");
            dimList.Add(value);
        }
        if (dimList.Count == 0 || parts.Length == 1)
            throw new InvalidDataException("NPY header has malformed shape tuple: " + path + ".");
        return dimList.ToArray();
    }

    static DenseTensor<T> ToTensor<T>(byte[] payload, int[] shape) where T : unmanaged
    {
        var values = new T[payload.Length / System.Runtime.InteropServices.Marshal.SizeOf<T>()];
        Buffer.BlockCopy(payload, 0, values, 0, payload.Length);
        return new DenseTensor<T>(values, shape);
    }
}
