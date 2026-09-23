using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

static class Driver
{
    const float Sentinel = -1234567.5f;
    static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };
    static string Hash(float[] a) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(a.AsSpan())));
    static void Require(bool value, string reason) { if (!value) throw new InvalidOperationException(reason); }
    sealed record Error(int Count, double Absolute, double Scaled, int WorstIndex, double Actual, double Reference, int DifferentBits, bool Passed);

    static Error Compare(float[] actual, double[] reference)
    {
        Require(actual.Length == reference.Length, "comparison extent");
        double max = 0, absolute = 0; int worst = 0, different = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            Require(float.IsFinite(actual[i]) && double.IsFinite(reference[i]), "nonfinite successful result");
            double d = Math.Abs(actual[i] - reference[i]);
            double error = d / Math.Max(1, Math.Abs(reference[i]));
            if (error > max) { max = error; worst = i; }
            absolute = Math.Max(absolute, d);
            if (BitConverter.SingleToInt32Bits(actual[i]) != BitConverter.SingleToInt32Bits((float)reference[i])) different++;
        }
        return new(actual.Length, absolute, max, worst, actual[worst], reference[worst], different, max <= 1e-4);
    }

    static Error Compare(float[] a, float[] b) => Compare(a, Array.ConvertAll(b, x => (double)x));

    static double[] DirectDouble(float[] input, float[] weights, int c, int m, int h, int w)
    {
        var output = new double[m * h * w];
        for (int oc = 0; oc < m; oc++)
        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            double value = 0;
            for (int ic = 0; ic < c; ic++)
            for (int ky = 0; ky < 3; ky++)
            for (int kx = 0; kx < 3; kx++)
            {
                int iy = y + ky - 1, ix = x + kx - 1;
                if ((uint)iy < (uint)h && (uint)ix < (uint)w)
                    value += (double)input[(ic * h + iy) * w + ix] * weights[((oc * c + ic) * 3 + ky) * 3 + kx];
            }
            output[(oc * h + y) * w + x] = value;
        }
        return output;
    }

    static double[] Epilogue(double[] source, float[] bias, float[] residual, int spatial, bool relu)
    {
        var result = (double[])source.Clone();
        for (int i = 0; i < result.Length; i++)
        {
            if (bias.Length != 0) result[i] += bias[i / spatial];
            if (residual.Length != 0) result[i] += residual[i];
            if (relu && result[i] < 0) result[i] = 0;
        }
        return result;
    }

    static float[] Candidate(float[] input, float[] prepared, float[] bias, float[] residual, int c, int m, int h, int w, int lanes, bool relu, bool expected = true)
    {
        Require(ConvBlockedSpatial.PlanWinograd(c, m, h, w, out int ni, out int np, out int no), "scratch plan");
        // Sentinels outside each span check overruns; inside spans expose incomplete writes.
        var vi = Enumerable.Repeat(Sentinel, ni + 2).ToArray();
        var vp = Enumerable.Repeat(Sentinel, np + 2).ToArray();
        var vo = Enumerable.Repeat(Sentinel, no + 2).ToArray();
        var result = Enumerable.Repeat(Sentinel, no + 2).ToArray();
        bool success = ConvBlockedSpatial.ExecuteWinograd(input, prepared, bias, residual, result.AsSpan(1, no),
            vi.AsSpan(1, ni), vp.AsSpan(1, np), vo.AsSpan(1, no), c, m, h, w, lanes, relu);
        Require(success == expected, "unexpected Winograd admission");
        foreach (var a in new[] { vi, vp, vo, result }) Require(a[0] == Sentinel && a[^1] == Sentinel, "scratch/output guard");
        if (!success) Require(result.All(x => x == Sentinel), "rejected destination changed");
        return result.AsSpan(1, no).ToArray();
    }

    static float[] Current(float[] input, float[] prepared, float[] bias, float[] residual, int c, int m, int h, int w, int lanes, bool relu)
    {
        var output = new float[m * h * w];
        Require(ConvBlockedSpatial.Execute(input, prepared, bias, residual, output,
            new float[c * (h + 2) * (w + 2)], new float[output.Length], c, m, h, w, 1, lanes, relu), "direct control refused");
        return output;
    }

    static IEnumerable<(int c, int m, int h, int w)> RawGeometries()
    {
        var pairs = new[] { (16,32), (16,48), (32,32), (32,64), (64,48), (64,64), (128,128), (256,256) };
        foreach (var shapes in new[] {
            new[] { (1,1), (1,7), (3,5), (4,8), (5,17), (6,16) },
            new[] { (5,33), (6,34), (5,65), (6,66) } })
        foreach (var (c, m) in pairs)
        foreach (var (h, w) in shapes)
            yield return (c, m, h, w);
    }

    static List<object> Raw(int lanes)
    {
        var rows = new List<object>(); int index = 0;
        foreach (var (c, m, h, w) in RawGeometries())
        foreach (string pattern in new[] { "random", "impulse", "cancellation" })
        {
            var random = new Random(1977 + c * 3 + m + h * 5 + w);
            var input = new float[c * h * w]; var weights = new float[m * c * 9];
            if (pattern == "random")
            {
                for (int i = 0; i < input.Length; i++) input[i] = random.Next(-127, 128) / 128f;
                for (int i = 0; i < weights.Length; i++) weights[i] = random.Next(-63, 64) / 256f;
            }
            else if (pattern == "impulse")
            {
                input[(h / 2) * w + w / 2] = 2;
                for (int oc = 0; oc < m; oc++) weights[oc * c * 9 + oc % 9] = .5f;
            }
            else
            {
                for (int i = 0; i < input.Length; i++) input[i] = (i % 3 - 1) * .125f;
                for (int i = 0; i < weights.Length; i++) weights[i] = (i % 2 == 0 ? 1 : -1) * .0625f;
            }
            string inputHash = Hash(input), weightsHash = Hash(weights);
            var prepared = ConvBlockedSpatial.PrepareWinograd(weights, c, m, lanes)!;
            Require(prepared != null, "finite prepare");
            var direct = ConvBlockedSpatial.Prepare(weights, c, m, lanes);
            string preparedHash = Hash(prepared), directHash = Hash(direct);
            var reference = DirectDouble(input, weights, c, m, h, w);
            for (int epilogue = 0; epilogue < 8; epilogue++)
            {
                float[] bias = (epilogue & 1) == 0 ? Array.Empty<float>() : Enumerable.Range(0, m).Select(i => (i % 5 - 2) / 16f).ToArray();
                float[] residual = (epilogue & 2) == 0 ? Array.Empty<float>() : Enumerable.Range(0, m * h * w).Select(i => (i % 7 - 3) / 32f).ToArray();
                bool relu = (epilogue & 4) != 0;
                string bHash = Hash(bias), rHash = Hash(residual);
                var wanted = Epilogue(reference, bias, residual, h * w, relu);
                var current = Current(input, direct, bias, residual, c, m, h, w, lanes, relu);
                var candidate = Candidate(input, prepared, bias, residual, c, m, h, w, lanes, relu);
                string held = Hash(candidate);
                var repeated = Candidate(input, prepared, bias, residual, c, m, h, w, lanes, relu);
                Require(Hash(candidate) == held && Hash(repeated) == held, "repeatability/held output");
                Require(Hash(bias) == bHash && Hash(residual) == rHash, "epilogue input mutation");
                var a = Compare(candidate, wanted); var b = Compare(current, wanted);
                Require(pattern != "impulse" || a.Absolute == 0, "impulse geometry/arithmetic");
                rows.Add(new { index = index++, c, m, h, w, pattern, epilogue, candidate = a, current = b,
                    selectedDifference = Compare(candidate, current), output = held, repeated = true, heldOutput = true, readonlyInputs = true, guards = true });
            }
            Require(Hash(input) == inputHash && Hash(weights) == weightsHash && Hash(prepared) == preparedHash && Hash(direct) == directHash, "raw input mutation");
        }
        Require(rows.Count == 1920, "raw census");
        return rows;
    }

    static float[] Load(string folder, JsonElement entry)
    {
        string name = entry.GetProperty("path").GetString()!;
        Require(Path.GetFileName(name) == name, "fixture path");
        byte[] bytes = File.ReadAllBytes(Path.Combine(folder, name));
        Require(bytes.Length == entry.GetProperty("bytes").GetInt32()
            && Convert.ToHexStringLower(SHA256.HashData(bytes)) == entry.GetProperty("sha256").GetString(), "fixture digest");
        int elements = 1; foreach (var n in entry.GetProperty("shape").EnumerateArray()) elements = checked(elements * n.GetInt32());
        Require(bytes.Length == checked(elements * 4), "fixture shape extent");
        return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
    }

    static List<object> Captured(string folder, int lanes)
    {
        using var spec = JsonDocument.Parse(File.ReadAllText(Path.Combine(folder, "result.json")));
        var rows = new List<object>();
        foreach (var call in spec.RootElement.GetProperty("calls").EnumerateArray())
        {
            var strides = call.GetProperty("attributes").GetProperty("strides").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            if (!call.GetProperty("eligible").GetBoolean() || !strides.SequenceEqual(new[] { 1, 1 })) continue;
            int[] shape = call.GetProperty("input").GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            int c = shape[1], h = shape[2], w = shape[3], m = call.GetProperty("weights").GetProperty("shape")[0].GetInt32();
            Require(shape[0] == 1, "batch");
            var input = Load(folder, call.GetProperty("input")); var weights = Load(folder, call.GetProperty("weights"));
            var bias = Load(folder, call.GetProperty("bias"));
            var residual = call.GetProperty("residual").ValueKind == JsonValueKind.Null ? Array.Empty<float>() : Load(folder, call.GetProperty("residual"));
            var native = Load(folder, call.GetProperty("output")); bool relu = call.GetProperty("relu").GetBoolean();
            var prepared = ConvBlockedSpatial.PrepareWinograd(weights, c, m, lanes)!;
            Require(prepared != null, "captured finite prepare");
            var direct = ConvBlockedSpatial.Prepare(weights, c, m, lanes);
            var inputs = new[] { input, weights, bias, residual, native, prepared, direct };
            string[] before = inputs.Select(Hash).ToArray();
            var current = Current(input, direct, bias, residual, c, m, h, w, lanes, relu);
            var candidate = Candidate(input, prepared, bias, residual, c, m, h, w, lanes, relu);
            string held = Hash(candidate);
            var repeated = Candidate(input, prepared, bias, residual, c, m, h, w, lanes, relu);
            Require(Hash(repeated) == held && Hash(candidate) == held, "captured repeatability/ownership");
            Require(inputs.Select(Hash).SequenceEqual(before), "captured readonly input");
            rows.Add(new { index = call.GetProperty("index").GetInt32(), fixture = call.GetProperty("case").GetString(),
                form = call.GetProperty("form").GetInt32(), c, m, h, w, relu,
                native = call.GetProperty("output").GetProperty("sha256").GetString(),
                candidate = Compare(candidate, native), current = Compare(current, native), selectedDifference = Compare(candidate, current),
                output = held, selectedOutput = Hash(current), repeated = true, heldOutput = true, readonlyInputs = true, guards = true,
                preparedBytes = prepared.Length * 4 });
            Console.WriteLine("captured " + rows.Count + "/87");
        }
        Require(rows.Count == 87, "captured census");
        return rows;
    }

    static int Refusals(int lanes)
    {
        const int c = 16, m = 32, h = 3, w = 7;
        int count = 0;
        foreach (float special in new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity })
        foreach (string operand in new[] { "input", "prepared", "bias", "residual", "weights" })
        {
            var input = new float[c * h * w]; var weights = new float[c * m * 9];
            var bias = new float[m]; var residual = new float[m * h * w];
            if (operand == "weights") { weights[3] = special; Require(ConvBlockedSpatial.PrepareWinograd(weights,c,m,lanes) == null, "weight refusal"); }
            else
            {
                var prepared = ConvBlockedSpatial.PrepareWinograd(weights,c,m,lanes)!;
                (operand switch { "input" => input, "prepared" => prepared, "bias" => bias, _ => residual })[3] = special;
                Candidate(input, prepared, bias, residual, c,m,h,w,lanes,true,false);
            }
            count++;
        }
        var huge = Enumerable.Repeat(float.MaxValue, c*m*9).ToArray();
        Require(ConvBlockedSpatial.PrepareWinograd(huge,c,m,lanes) == null, "weight-transform overflow refusal"); count++;
        foreach (var (x,wv,b,r) in new[] { (float.MaxValue,1f,0f,0f), (1e20f,1e20f,0f,0f), (1f,1f,float.MaxValue,0f), (1f,1f,0f,float.MaxValue) })
        {
            var input = Enumerable.Repeat(x,c*h*w).ToArray();
            var prepared = ConvBlockedSpatial.PrepareWinograd(Enumerable.Repeat(wv,c*m*9).ToArray(),c,m,lanes)!;
            Candidate(input,prepared,Enumerable.Repeat(b,m).ToArray(),Enumerable.Repeat(r,m*h*w).ToArray(),c,m,h,w,lanes,true,false); count++;
        }
        Require(!ConvBlockedSpatial.PlanWinograd(256,256,int.MaxValue,int.MaxValue,out _,out _,out _), "oversized scratch plan"); count++;
        return count;
    }

    static int Contracts(int lanes)
    {
        const int c=16, m=32, h=3, w=7;
        int[] lengths = { c*h*w, 16*c*m, m, m*h*w, m*h*w, 16*c*8, 16*m*8, m*h*w };
        var pairs = new List<(int,int)>();
        for (int read=0; read<4; read++) for (int write=4; write<8; write++) pairs.Add((read,write));
        for (int left=4; left<8; left++) for (int right=left+1; right<8; right++) pairs.Add((left,right));
        pairs.Add((-1,0)); pairs.Add((-1,4));
        foreach (var (left,right) in pairs)
        {
            var arrays = Enumerable.Range(0,8).Select(_ => Enumerable.Repeat(Sentinel,lengths.Max()+2).ToArray()).ToArray();
            if (left>=0) arrays[right]=arrays[left];
            int[] sizes=(int[])lengths.Clone();if (left<0) sizes[right]--;
            string[] before=arrays.Select(Hash).ToArray();bool rejected=false;
            try
            {
                ConvBlockedSpatial.ExecuteWinograd(arrays[0].AsSpan(1,sizes[0]),arrays[1].AsSpan(1,sizes[1]),
                    arrays[2].AsSpan(1,sizes[2]),arrays[3].AsSpan(1,sizes[3]),arrays[4].AsSpan(1,sizes[4]),
                    arrays[5].AsSpan(1,sizes[5]),arrays[6].AsSpan(1,sizes[6]),arrays[7].AsSpan(1,sizes[7]),c,m,h,w,lanes,true);
            }
            catch (ArgumentException) { rejected=true; }
            Require(rejected && arrays.Select(Hash).SequenceEqual(before), "extent/alias validation before mutation");
        }
        Require(pairs.Count==24,"contract census");return pairs.Count;
    }

    delegate bool RangeGuard(ReadOnlySpan<float> values);

    static int RangeCase(RangeGuard check, float[] values, int offset, int length)
    {
        var span = values.AsSpan(offset, length);
        bool expected = true;
        foreach (float value in span) if (MathF.Abs(value) > float.MaxValue / 4) expected = false;
        string before = Hash(values);
        Require(check(span) == expected, "range guard scalar agreement");
        Require(Hash(values) == before, "range guard input mutation");
        return 1;
    }

    static int RangeGuards()
    {
        var method = typeof(ConvBlockedSpatial).GetMethod("EpilogueRange",
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!;
        var check = method.CreateDelegate<RangeGuard>();
        const float limit = float.MaxValue / 4;
        float below = MathF.BitDecrement(limit), above = MathF.BitIncrement(limit);
        float[] cases = { 0f, BitConverter.Int32BitsToSingle(int.MinValue), float.Epsilon, -float.Epsilon,
            below, limit, above, -below, -limit, -above, float.MaxValue, -float.MaxValue,
            float.PositiveInfinity, float.NegativeInfinity, float.NaN };
        int count = 0;
        foreach (int length in new[] { 0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65 })
        foreach (int offset in new[] { 0, 1, 3 })
        {
            // Disallowed magnitude outside the span exposes a vector overread on all-zero baselines.
            var values = Enumerable.Repeat(float.MaxValue, offset + length + 17).ToArray();
            values.AsSpan(offset, length).Clear();
            count += RangeCase(check, values, offset, length);
            foreach (float value in cases)
            for (int index = 0; index < length; index++)
            {
                values[offset + index] = value;
                count += RangeCase(check, values, offset, length);
                values[offset + index] = 0f;
            }
        }
        Require(count == 10566, "range guard census");
        return count;
    }

    static int Main(string[] args)
    {
        Require(args.Length == 4, "mode width fixtures result");
        string mode = args[0]; int width = int.Parse(args[1]), lanes = width / 32;
        Require((width == 256 || width == 512) && Avx2.IsSupported && Fma.IsSupported && Avx512F.IsSupported == (width == 512), "ISA identity");
        Require(Environment.Version.ToString() == "10.0.8", "runtime identity");
        var rows = mode == "raw" ? Raw(lanes) : mode == "captured" ? Captured(args[2], lanes) : throw new ArgumentException("mode");
        int refusals = mode == "raw" ? Refusals(lanes) : 0;
        int contracts = mode == "raw" ? Contracts(lanes) : 0;
        int rangeChecks = mode == "raw" ? RangeGuards() : 0;
        // Keep every numerical failure in a terminal report; external audit decides admission.
        File.WriteAllText(args[3], JsonSerializer.Serialize(new { mode, width, runtime = Environment.Version.ToString(),
            pid = Environment.ProcessId, assembly = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(Driver).Assembly.Location))),
            rows, refusals, contracts, rangeChecks, noPerformanceMeasurement = true, completed = true }, JsonOptions) + "\n");
        return 0;
    }
}
