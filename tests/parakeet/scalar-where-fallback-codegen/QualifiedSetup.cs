using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

static partial class Program
{
    static readonly JsonSerializerOptions Json = new() { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower, WriteIndented = true };
    static readonly uint[] FloatBits = [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321, 0x7f812345, 1, 0x807fffff, 0x3f800000, 0xc61c4000];
    static string Base = "", Role = "", Mode = "";
    static readonly List<object> Rows = [];
    static void Require(bool value, string message) { if (!value) throw new InvalidDataException(message); }
    static string Hash(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
    static string FileHash(string path) => Hash(File.ReadAllBytes(path));
    static int[] Shape(JsonElement e, string name) => e.GetProperty(name).EnumerateArray().Select(v => v.GetInt32()).ToArray();
    static string Text(JsonElement e, string name, string fallback = "") => e.TryGetProperty(name, out var v) && v.ValueKind != JsonValueKind.Null ? v.GetString()! : fallback;
    static int Count(int[] shape) => shape.Aggregate(1, (a, b) => checked(a * b));
    static byte[] Bytes<T>(T[] array) where T : unmanaged => MemoryMarshal.AsBytes(array.AsSpan()).ToArray();
    sealed record Input<T>(Tensor<T> Tensor, T[] Store) where T : unmanaged;
    sealed class SemanticDense<T>(Memory<T> data, int[] shape) : DenseTensor<T>(data, shape) where T : unmanaged
    {
        public override T GetValue(int index)
        {
            var value = base.GetValue(index);
            if (typeof(T) == typeof(bool)) return (T)(object)!(bool)(object)value;
            if (typeof(T) == typeof(float)) return (T)(object)BitConverter.Int32BitsToSingle(BitConverter.SingleToInt32Bits((float)(object)value) ^ int.MinValue);
            return value;
        }
    }
    static T[] Values<T>(int length, int seed) where T : unmanaged
    {
        var values = new T[length]; var bytes = MemoryMarshal.AsBytes(values.AsSpan());
        for (int i = 0; i < bytes.Length; i++) bytes[i] = typeof(T) == typeof(bool) ? (byte)((i + seed) % 2) : (byte)((i * 17 + seed * 31) % 256);
        if (typeof(T) == typeof(float))
        {
            var bits = MemoryMarshal.Cast<T, uint>(values.AsSpan());
            for (int i = 0; i < length; i++) bits[i] = FloatBits[(i + seed) % FloatBits.Length];
        }
        return values;
    }
    static Input<T> Make<T>(T[] data, int[] shape, string layout, string operand) where T : unmanaged
    {
        if (layout == operand + "-view")
        {
            var parentShape = (int[])shape.Clone(); parentShape[^1] *= 2;
            var store = Values<T>(Count(parentShape), 9);
            for (int i = 0; i < data.Length; i++) store[2 * i + 1] = data[i];
            var parent = new DenseTensor<T>(store.AsMemory(), parentShape);
            var slices = shape.Select(_ => SliceIndex.All).ToArray(); slices[^1] = new SliceIndex(1, parentShape[^1], 2);
            return new(new TensorSlice<T>(parent, slices), store);
        }
        int offset = layout == "offset" ? 3 : 0;
        var backing = Values<T>(data.Length + 2 * offset, 10); data.CopyTo(backing, offset);
        Tensor<T> tensor = layout == operand + "-custom" ? new SemanticDense<T>(backing.AsMemory(offset, data.Length), shape)
            : new DenseTensor<T>(backing.AsMemory(offset, data.Length), shape, layout == operand + "-reverse");
        if (layout == "condition-broadcast" && operand == "condition")
        {
            backing = [data[0]];
            tensor = Tensor<T>.BroadcastTo(new DenseTensor<T>(backing.AsMemory(), new int[shape.Length].Select(_ => 1).ToArray()), shape);
        }
        return new(tensor, backing);
    }
    // Independent right-aligned coordinate oracle; it never calls Broadcast/Where.
    static int[] OutputShape(params int[][] shapes)
    {
        int rank = shapes.Max(s => s.Length); var output = Enumerable.Repeat(1, rank).ToArray();
        foreach (var shape in shapes)
            for (int i = 0; i < shape.Length; i++)
            {
                int axis = rank - shape.Length + i, a = output[axis], b = shape[i];
                if (a != b && a != 1 && b != 1) throw new ArgumentException("oracle incompatible shapes");
                if (a == 1) output[axis] = b;
            }
        return output;
    }
    static int[] Coordinates(int index, int[] shape)
    {
        var coords = new int[shape.Length];
        for (int i = shape.Length - 1; i >= 0; i--) { coords[i] = index % shape[i]; index /= shape[i]; }
        return coords;
    }
    static T At<T>(Tensor<T> tensor, int[] coords) where T : unmanaged
    {
        int index = 0, offset = coords.Length - tensor.Rank;
        for (int i = 0; i < tensor.Rank; i++) index += (tensor.Dimensions[i] == 1 ? 0 : coords[offset + i]) * tensor.Strides[i];
        return tensor.GetValue(index);
    }
    static T[] Logical<T>(Tensor<T> tensor) where T : unmanaged
    {
        if (tensor.GetType() == typeof(DenseTensor<T>) && !tensor.IsReversedStride) return ((DenseTensor<T>)tensor).Buffer.ToArray();
        var shape = tensor.Dimensions.ToArray(); var values = new T[Count(shape)];
        for (int i = 0; i < values.Length; i++) values[i] = At(tensor, Coordinates(i, shape));
        return values;
    }
    static T[] Oracle<T>(Tensor<bool> c, Tensor<T> x, Tensor<T> y, out int[] shape) where T : unmanaged
    {
        shape = OutputShape(c.Dimensions.ToArray(), x.Dimensions.ToArray(), y.Dimensions.ToArray());
        var values = new T[Count(shape)];
        for (int i = 0; i < values.Length; i++)
        {
            var coords = Coordinates(i, shape);
            values[i] = At(c, coords) ? At(x, coords) : At(y, coords);
        }
        return values;
    }
    static void Mutate<T>(T[] store) where T : unmanaged
    {
        var bytes = MemoryMarshal.AsBytes(store.AsSpan()); if (bytes.Length != 0) bytes[0] ^= 1;
    }
    static void Run<T>(JsonElement spec) where T : unmanaged
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
        var before = new[] { Hash(Bytes(ci.Store)), Hash(Bytes(xi.Store)), Hash(Bytes(yi.Store)) };
        string[] Stores() => [Hash(Bytes(ci.Store)), Hash(Bytes(xi.Store)), Hash(Bytes(yi.Store))];
        bool? admitted = null;
        if (Role == "candidate" && typeof(T) == typeof(float))
        {
            var method = typeof(Tensor<float>).Assembly.GetType("Lokad.Onnx.UniformScalarWhere", true)!.GetMethod("Try", BindingFlags.Static | BindingFlags.NonPublic)!.MakeGenericMethod(typeof(T));
            object?[] args = [c, x, y, null]; admitted = (bool)method.Invoke(null, args)!;
            Require(admitted == spec.GetProperty("eligible").GetBoolean(), name + " helper admission");
            if (admitted == false) Require(args[3] is null, name + " refused output");
            else Require(Bytes(Logical((Tensor<T>)args[3]!)).SequenceEqual(Bytes(Oracle(c, x, y, out _))), name + " helper oracle");
            Require(before.SequenceEqual(Stores()), name + " helper changed inputs");
        }
        string expectedError = Text(spec, "error");
        if (expectedError != "")
        {
            Exception? error = null;
            try { Tensor<T>.Where(c, x, y); } catch (Exception e) { error = e; }
            Require(error?.GetType().FullName == expectedError, name + " error " + error);
            Require(before.SequenceEqual(Stores()), name + " error changed inputs");
            Rows.Add(new { name, dtype, error = expectedError, inputs = true, helper_admitted = admitted }); return;
        }
        var expected = Oracle(c, x, y, out var shape); var output = Tensor<T>.Where(c, x, y);
        Require(shape.SequenceEqual(output.Dimensions.ToArray()) && Bytes(Logical(output)).SequenceEqual(Bytes(expected)), name + " coordinate oracle");
        Require(before.SequenceEqual(Stores()), name + " changed inputs");
        var held = Bytes(Logical(output)); var outputHash = Hash(held);
        if (spec.TryGetProperty("reference", out var reference))
            Require(held.SequenceEqual(File.ReadAllBytes(Path.Combine(Base, "fixtures", reference.GetString()!))), name + " captured reference");
        var codegen = Mode == "codegen";
        if (codegen)
        {
            for (int i = 0; i < 80; i++) Require(Hash(Bytes(Logical(Tensor<T>.Where(c, x, y)))) == outputHash, name + " codegen output");
            Rows.Add(new { name, dtype, values = expected.Length, shape, output = outputHash, calls = 81 }); return;
        }
        Mutate(ci.Store); Mutate(xi.Store); if (!ReferenceEquals(xi.Store, yi.Store)) Mutate(yi.Store);
        var changedInputs = Stores(); var changedExpected = Oracle(c, x, y, out _); var second = Tensor<T>.Where(c, x, y);
        Require(Bytes(Logical(second)).SequenceEqual(Bytes(changedExpected)), name + " mutated oracle");
        Require(Bytes(Logical(output)).SequenceEqual(held), name + " held output aliases input");
        Require(changedInputs.SequenceEqual(Stores()), name + " changed inputs in second call");
        string mutatedOutput = Hash(Bytes(Logical(second)));
        if (output.Length != 0) output.SetValue(0, Values<T>(1, 8)[0]);
        Require(changedInputs.SequenceEqual(Stores()) && Hash(Bytes(Logical(second))) == mutatedOutput, name + " output aliases input/other output");
        Rows.Add(new { name, dtype, values = expected.Length, shape, output = outputHash, mutated_output = mutatedOutput,
            oracle = true, inputs = true, held = true, owned = true, helper_admitted = admitted });
    }
    static int NumericalMain(string[] args)
    {
        Require(args.Length == 5, "arguments"); Base = args[0]; Role = args[1]; Mode = args[2]; int width = int.Parse(args[3]);
        Require(Environment.Version.ToString() == "10.0.8", "runtime");
        using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(Base, "cases.json")));
        foreach (var spec in manifest.RootElement.EnumerateArray())
        {
            if (Mode == "codegen" && !Text(spec, "name").StartsWith("capture-")) continue;
            switch (Text(spec, "dtype"))
            {
                case "Float": Run<float>(spec); break; case "Bool": Run<bool>(spec); break;
                case "Byte": Run<byte>(spec); break; case "Int32": Run<int>(spec); break;
                case "Int64": Run<long>(spec); break; case "UInt32": Run<uint>(spec); break;
                case "UInt64": Run<ulong>(spec); break; case "Double": Run<double>(spec); break;
                case "Half": Run<Half>(spec); break; default: throw new InvalidDataException("dtype");
            }
        }
        var flags = Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>()
            .Where(v => ((string)v.Key).StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || ((string)v.Key).StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase) || ((string)v.Key).StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase))
            .ToDictionary(v => (string)v.Key, v => (string)v.Value!);
        File.WriteAllText(args[4], JsonSerializer.Serialize(new { completed = true, no_performance_measurement = true,
            pid = Environment.ProcessId, runtime = Environment.Version.ToString(), role = Role, mode = Mode, width, flags,
            assembly = FileHash(Assembly.GetExecutingAssembly().Location), core_sha256 = FileHash(typeof(Tensor<float>).Assembly.Location),
            avx512 = System.Runtime.Intrinsics.X86.Avx512F.IsSupported, rows = Rows }, Json));
        return 0;
    }
}
