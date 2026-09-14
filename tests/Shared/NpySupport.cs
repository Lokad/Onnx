using Lokad.Onnx;

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
        byte[] raw = File.ReadAllBytes(path);
        if (raw.Length < 10)
            throw new InvalidDataException("Truncated NPY file: " + path + ".");
        byte[] magic = new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' };
        for (int i = 0; i < magic.Length; i++)
            if (raw[i] != magic[i])
                throw new InvalidDataException("Bad NPY magic: " + path + ".");
        int major = raw[6];
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
            headerLen = raw[8] | (raw[9] << 8) | (raw[10] << 16) | (raw[11] << 24);
        }
        else throw new InvalidDataException("Unsupported NPY version " + major + ": " + path + ".");
        if (headerLen < 0 || headerOff + headerLen > raw.Length)
            throw new InvalidDataException("NPY header overruns file: " + path + ".");
        string header = System.Text.Encoding.ASCII.GetString(raw, headerOff, headerLen);
        string kind = ParseDescr(header, path);
        if (header.Contains("'fortran_order': True"))
            throw new InvalidDataException("Fortran-order NPY is not supported: " + path + ".");
        if (!header.Contains("'fortran_order': False"))
            throw new InvalidDataException("NPY header lacks fortran_order: " + path + ".");
        int[] shape = ParseShape(header, path);
        long n = 1;
        foreach (int d in shape)
        {
            if (d < 0)
                throw new InvalidDataException("Negative NPY dimension: " + path + ".");
            n = checked(n * d);
        }
        int size = kind == "<i8" ? 8 : 4;
        long expected = checked(n * size);
        long actual = raw.Length - headerOff - headerLen;
        if (actual != expected)
            throw new InvalidDataException("NPY payload size mismatch: " + path + " (expected " + expected + " payload bytes, file holds " + actual + ").");
        var payload = new byte[expected];
        Array.Copy(raw, headerOff + headerLen, payload, 0, payload.Length);
        return (kind, shape, payload);
    }

    static string ParseDescr(string header, string path)
    {
        int key = header.IndexOf("'descr'", StringComparison.Ordinal);
        if (key < 0) throw new InvalidDataException("NPY header lacks descr: " + path + ".");
        int sq = header.IndexOf((char)39, key + 7);
        int dq = header.IndexOf((char)34, key + 7);
        int quote = -1;
        char q = (char)0;
        if (sq >= 0 && (dq < 0 || sq < dq)) { quote = sq; q = (char)39; }
        else if (dq >= 0) { quote = dq; q = (char)34; }
        if (quote < 0) throw new InvalidDataException("NPY header has malformed descr: " + path + ".");
        int end = header.IndexOf(q, quote + 1);
        if (end < 0) throw new InvalidDataException("NPY header has unterminated descr: " + path + ".");
        string token = header.Substring(quote + 1, end - quote - 1);
        if (token == "<f4" || token == "<i4" || token == "<i8") return token;
        throw new InvalidDataException("Unsupported NPY dtype " + token + ": " + path + ".");
    }

    static int[] ParseShape(string header, string path)
    {
        int key = header.IndexOf("'shape'", StringComparison.Ordinal);
        if (key < 0) throw new InvalidDataException("NPY header lacks shape: " + path + ".");
        int open = header.IndexOf('(', key);
        int close = open >= 0 ? header.IndexOf(')', open + 1) : -1;
        if (open < 0 || close < 0) throw new InvalidDataException("NPY header has malformed shape: " + path + ".");
        string inner = header.Substring(open + 1, close - open - 1).Trim();
        if (inner.Length == 0) return Array.Empty<int>();
        string[] parts = inner.Split(',');
        var dimList = new System.Collections.Generic.List<int>();
        for (int i = 0; i < parts.Length; i++)
        {
            string part = parts[i].Trim();
            if (part.Length == 0) continue;
            int value;
            try { value = int.Parse(part, System.Globalization.CultureInfo.InvariantCulture); }
            catch (Exception ex) { throw new InvalidDataException("NPY header has non-integer shape: " + path + ".", ex); }
            dimList.Add(value);
        }
        if (dimList.Count == 0) throw new InvalidDataException("NPY header has malformed shape: " + path + ".");
        return dimList.ToArray();
    }

    static DenseTensor<T> ToTensor<T>(byte[] payload, int[] shape) where T : unmanaged
    {
        var values = new T[payload.Length / System.Runtime.InteropServices.Marshal.SizeOf<T>()];
        Buffer.BlockCopy(payload, 0, values, 0, payload.Length);
        return new DenseTensor<T>(values, shape);
    }
}
