using System.Runtime.InteropServices;
using System.Text.Json;
using Lokad.Onnx;

namespace LayerNormOutput;

internal static partial class Program
{
    static void Proof(string capture, string output)
    {
        Require(!Directory.Exists(output), "Proof output exists"); Directory.CreateDirectory(output);
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(capture, "capture.json")));
        var metadata = document.RootElement; Require(metadata.GetProperty("passed").GetBoolean(), "Capture failed");
        Require(metadata.GetProperty("identity").GetProperty("core_sha256").GetString() == Core, "Capture core");
        foreach (var entry in metadata.GetProperty("files").EnumerateObject())
        {
            string path = Path.Combine(capture, entry.Name);
            Require(new FileInfo(path).Length == entry.Value.GetProperty("bytes").GetInt64() && Hash(path) == entry.Value.GetProperty("sha256").GetString(), "Changed capture " + entry.Name);
        }
        long comparisons = 0; int count = 0; double maxScalarError = 0;
        var records = new List<object>();
        var variants = new (string Name, Kernel Run)[] { ("copy", Kernels.Copy), ("wide", Kernels.WideOutput) };
        void Check(string name, float[] input, float[] scale, float[]? bias, int block, int outer, float epsilon, string? savePrefix)
        {
            Require(input.Length == block * outer && scale.Length == block && (bias is null || bias.Length == block), "Proof geometry");
            const float sentinel = -12345.5f;
            float[] xBacking = Enumerable.Repeat(sentinel, input.Length + 13).ToArray(); input.CopyTo(xBacking, 3);
            float[] sBacking = Enumerable.Repeat(sentinel, block + 11).ToArray(); scale.CopyTo(sBacking, 5);
            float[] bBacking = Enumerable.Repeat(sentinel, block + 9).ToArray(); bias?.CopyTo(bBacking, 2);
            var xt = new DenseTensor<float>(xBacking.AsMemory(3, input.Length), new[] { outer, block });
            var st = new DenseTensor<float>(sBacking.AsMemory(5, block), new[] { block });
            var bt = bias is null ? null : new DenseTensor<float>(bBacking.AsMemory(2, block), new[] { block });
            string xHash = Hash(xBacking), sHash = Hash(sBacking), bHash = Hash(bBacking);
            var expected = new DenseTensor<float>(new[] { outer, block }); Product(xt, st, bt, expected, block, outer, epsilon);
            var want = expected.ToArray(); double scalarError = ScalarError(input, scale, bias, want, block, outer, epsilon);
            Require(scalarError <= 1e-5, "Independent scalar reference " + name); maxScalarError = Math.Max(maxScalarError, scalarError);
            Require(Hash(xBacking) == xHash && Hash(sBacking) == sHash && Hash(bBacking) == bHash, "Product changed input or parameters");
            foreach (var variant in variants)
            {
                var backing = Enumerable.Repeat(sentinel, input.Length + 17).ToArray(); var y = new DenseTensor<float>(backing.AsMemory(7, input.Length), new[] { outer, block });
                variant.Run(xt, st, bt, y, block, outer, epsilon);
                Require(MemoryMarshal.AsBytes(y.Buffer.Span).SequenceEqual(MemoryMarshal.AsBytes(want.AsSpan())), variant.Name + " bits: " + name);
                Require(backing.Take(7).Concat(backing.Skip(7 + input.Length)).All(v => v == sentinel), "Destination sentinel " + name);
                var inBacking = xBacking.ToArray(); var inplace = new DenseTensor<float>(inBacking.AsMemory(3, input.Length), new[] { outer, block });
                variant.Run(inplace, st, bt, inplace, block, outer, epsilon);
                Require(MemoryMarshal.AsBytes(inplace.Buffer.Span).SequenceEqual(MemoryMarshal.AsBytes(want.AsSpan())), variant.Name + " in-place bits: " + name);
                Require(inBacking.Take(3).Concat(inBacking.Skip(3 + input.Length)).All(v => v == sentinel), "In-place sentinel " + name);
                Require(Hash(xBacking) == xHash && Hash(sBacking) == sHash && Hash(bBacking) == bHash, "Input/parameter mutation " + name);
                comparisons += 2L * want.Length;
                if (savePrefix is not null) Save(Path.Combine(output, savePrefix + "-" + variant.Name + ".f32"), y.ToArray());
            }
            count++; records.Add(new { name, block, outer, epsilon, has_bias = bias is not null, values = want.Length, input_sha256 = Hash(input), scale_sha256 = Hash(scale),
                bias_sha256 = bias is null ? null : Hash(bias), product_sha256 = Hash(want), scalar_error = scalarError, exact = true, guards = true, inputs_preserved = true, inplace = true, prefix = savePrefix });
        }
        uint state = 20260920;
        uint Bits() { state = unchecked(state * 1664525 + 1013904223); return state; }
        float Finite() => (int)(Bits() >> 8) / 1048576f - 8;
        int[] widths = [1,2,3,4,7,8,9,15,16,17,23,24,25,31,32,33,63,64,65,383,384,385,1024,1536];
        foreach (int width in widths)
        foreach (int rows in new[] { 0, 1, 2, 7 })
        foreach (float epsilon in new[] { 0f, 1e-12f, 1e-5f })
        foreach (bool bias in new[] { false, true })
            Check($"shape:{width}:{rows}:{epsilon}:{bias}", Enumerable.Range(0, width * rows).Select(_ => Finite()).ToArray(), Enumerable.Range(0, width).Select(_ => Finite()).ToArray(),
                bias ? Enumerable.Range(0, width).Select(_ => Finite()).ToArray() : null, width, rows, epsilon, null);
        float[] exceptional = [0f,-0f,float.Epsilon,-float.Epsilon,float.PositiveInfinity,float.NegativeInfinity,float.NaN,
            BitConverter.Int32BitsToSingle(0x7fa12345),BitConverter.Int32BitsToSingle(unchecked((int)0xffa54321)),float.MaxValue,-float.MaxValue];
        foreach (float value in exceptional)
        foreach (int width in new[] { 17, 32, 65 })
        {
            Check("uniform-exceptional:" + BitConverter.SingleToInt32Bits(value) + ":" + width, Enumerable.Repeat(value, 2 * width).ToArray(), Enumerable.Repeat(1f, width).ToArray(), null, width, 2, 1e-5f, null);
            Check("bias-exceptional:" + BitConverter.SingleToInt32Bits(value) + ":" + width, Enumerable.Range(0, 2 * width).Select(_ => Finite()).ToArray(), Enumerable.Repeat(1f, width).ToArray(), Enumerable.Repeat(value, width).ToArray(), width, 2, 1e-5f, null);
        }
        foreach (float center in new[] { 0f, 1e-20f, 1f, 1e6f, 1e20f })
        foreach (int width in new[] { 16, 17, 384, 385 })
            Check("nearly-constant:" + center + ":" + width, Enumerable.Range(0, 3 * width).Select(i => i % 3 == 0 ? float.BitIncrement(center) : center).ToArray(),
                Enumerable.Repeat(1f, width).ToArray(), Enumerable.Repeat(-0f, width).ToArray(), width, 3, 1e-12f, null);
        for (int index = 0; index < 128; index++)
        {
            int width = widths[index % widths.Length];
            Check("random-bits:" + index, Enumerable.Range(0, width * 2).Select(_ => BitConverter.Int32BitsToSingle(unchecked((int)Bits()))).ToArray(),
                Enumerable.Range(0, width).Select(_ => Finite()).ToArray(), index % 2 == 0 ? null : Enumerable.Range(0, width).Select(_ => Finite()).ToArray(), width, 2, 1e-5f, null);
        }
        int captured = 0;
        foreach (string name in Cases)
        {
            string folder = Path.Combine(capture, name); using var caseDocument = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(folder, "capture.json")));
            foreach (var node in caseDocument.RootElement.GetProperty("nodes").EnumerateArray())
            {
                int index = node.GetProperty("index").GetInt32(); string prefix = index.ToString("D2");
                float[] x = ReadFloats(Path.Combine(folder, prefix + "-x.f32")), scale = ReadFloats(Path.Combine(folder, prefix + "-scale.f32"));
                float[]? bias = node.GetProperty("has_bias").GetBoolean() ? ReadFloats(Path.Combine(folder, prefix + "-bias.f32")) : null;
                Check(name + ":" + index, x, scale, bias, node.GetProperty("block").GetInt32(), node.GetProperty("outer").GetInt32(), node.GetProperty("epsilon").GetSingle(), name + "-" + prefix);
                Require(Hash(Path.Combine(output, name + "-" + prefix + "-wide.f32")) == Hash(Path.Combine(folder, prefix + "-y.f32")), "Candidate/captured graph output bits");
                captured++;
            }
        }
        Require(captured == 125, "Complete real capture coverage");
        Write(Path.Combine(output, "proof.json"), new { passed = true, identity = Identity(), capture_sha256 = Hash(Path.Combine(capture, "capture.json")), cases = count, comparisons,
            captured, maximum_scalar_error = maxScalarError, records, files = Inventory(output) });
        Console.WriteLine($"Exact LayerNorm proof: {count} cases, {comparisons} comparisons, {captured} captures; scalar error {maxScalarError}");
    }

    static double ScalarError(float[] x, float[] scale, float[]? bias, float[] actual, int block, int outer, float epsilon)
    {
        double maximum = 0;
        for (int row = 0; row < outer; row++)
        {
            int offset = row * block; double mean = 0;
            for (int i = 0; i < block; i++) mean += x[offset + i]; mean /= block;
            double variance = 0;
            for (int i = 0; i < block; i++) { double d = x[offset + i] - mean; variance += d * d; }
            double inv = 1 / Math.Sqrt(variance / block + epsilon);
            for (int i = 0; i < block; i++)
            {
                float want = (float)(((x[offset + i] - mean) * inv) * scale[i] + (bias is null ? 0f : bias[i]));
                float got = actual[offset + i];
                if (float.IsNaN(want)) Require(float.IsNaN(got), "Scalar NaN class");
                else if (float.IsInfinity(want)) Require(want == got, "Scalar infinity class");
                else { Require(float.IsFinite(got), "Scalar finite class"); maximum = Math.Max(maximum, Math.Abs((double)want - got) / Math.Max(1, Math.Abs((double)want))); }
            }
        }
        return maximum;
    }
}
