using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;

namespace Lokad.Onnx.Backend.Tests;

// Portable regression coverage extracted from the closed prepared-row contracts.
// The real Parakeet fixture and runtime/product guards remain in that campaign.
public unsafe class PreparedSingleRowTests
{
    delegate void PackedKernel(int n, int k, float* a, float* packed, float* output);
    const int Guard = 16;
    static readonly float Sentinel = BitConverter.Int32BitsToSingle(0x4a654321);
    static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
    static string Hash(ReadOnlySpan<float> values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values)));
    static float[] Values(int length, int seed) => Enumerable.Range(0, length).Select(i => ((i * 37 + seed) % 101 - 50) * .03125f).ToArray();
    static float[] Guarded(float[] values)
    {
        var result = Enumerable.Repeat(Sentinel, values.Length + 2 * Guard).ToArray();
        values.CopyTo(result, Guard); return result;
    }
    static void Guards(float[] values)
    {
        Require(values.AsSpan(0, Guard).ToArray().All(v => v == Sentinel), "Leading guard");
        Require(values.AsSpan(values.Length - Guard).ToArray().All(v => v == Sentinel), "Trailing guard");
    }
    static void Equal(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string label)
    {
        Require(expected.Length == actual.Length, label + " length");
        for (int i = 0; i < expected.Length; i++)
            Require(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]), label + " bits at " + i);
    }
    static float[] Pack(float[] b, int n, int k)
    {
        var packed = new float[b.Length];
        fixed (float* source = b, target = packed) MathOps.PackPanelsB(n, k, source, target);
        return packed;
    }
    static Dictionary<float[], PackedMatMulWeight> Mapping(DenseTensor<float> weight, float[] array)
    {
        int n = weight.Dimensions[0], k = weight.Dimensions[1];
        var packed = new DenseTensor<float>(Pack(array, n, k), new[] { n, k });
        packed.Name = "packed:w";
        return new() { [array] = new PackedMatMulWeight("w", weight, array.Length, array, "packed:w", packed) };
    }
    static void Raw(string name, int n, int k, float[] aValues, float[] bValues, float[] initial, PackedKernel kernel)
    {
        var a = Guarded(aValues); var b = Guarded(bValues);
        var packed = Guarded(new float[n * k]);
        var expected = Guarded(initial); var actual = Guarded(initial);
        string aBefore = Hash(a), bBefore = Hash(b), packedBefore;
        fixed (float* ap = a, bp = b, pp = packed, ep = expected, cp = actual)
        {
            MathOps.PackPanelsB(n, k, bp + Guard, pp + Guard);
            packedBefore = Hash(packed);
            MathOps.mm_m1_kblocked(1, n, k, ap + Guard, bp + Guard, ep + Guard);
            kernel(n, k, ap + Guard, pp + Guard, cp + Guard);
        }
        Equal(expected, actual, name); Guards(a); Guards(b); Guards(packed); Guards(actual);
        Require(Hash(a) == aBefore && Hash(b) == bBefore, name + " immutable raw inputs");
        Require(Hash(packed) == packedBefore, name + " first packed input");
        long allocated;
        fixed (float* ap = a, pp = packed, cp = actual)
        {
            // Warm the delegate itself before checking the kernel's allocation contract.
            kernel(n, k, ap + Guard, pp + Guard, cp + Guard);
            long start = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 8; i++) kernel(n, k, ap + Guard, pp + Guard, cp + Guard);
            allocated = GC.GetAllocatedBytesForCurrentThread() - start;
        }
        Require(allocated == 0 && Hash(packed) == packedBefore && Hash(a) == aBefore, name + " raw allocation/ownership");
        Guards(actual);
    }
    static void Public(string name, DenseTensor<float> x, DenseTensor<float> b,
        IReadOnlyDictionary<float[], PackedMatMulWeight>? map, TensorExecutionOptions mode)
    {
        string input = Hash(x.Buffer.Span), weight = Hash(b.Buffer.Span);
        var packedBefore = map?.ToDictionary(p => p.Key, p => Hash(p.Value.Packed.Buffer.Span));
        var copy = new CopyAccountant(); var scratch = new ScratchAccountant();
        var options = mode with { PackedMatMulWeights = map, CopyReporter = copy, ScratchReporter = scratch };
        var reference = Tensor<float>.MatMul(x, b, mode with { PackedMatMulWeights = null }).ToArray();
        var got = Tensor<float>.MatMul(x, b, options);
        float[] held = got.ToArray(); Equal(reference, held, name);
        long copies = copy.TotalCopyBytes, scratches = scratch.TotalScratchBytes;
        var repeated = Tensor<float>.MatMul(x, b, options);
        Equal(held, got.ToArray(), name + " held output"); Equal(reference, repeated.ToArray(), name + " repeated output");
        Require(!TensorAlias.SharesBackingMemory(got, x) && !TensorAlias.SharesBackingMemory(got, b)
            && !TensorAlias.SharesBackingMemory(got, repeated), name + " owned outputs");
        var destination = new DenseTensor<float>(Enumerable.Repeat(17.25f, held.Length).ToArray(), got.Dimensions);
        Require(ReferenceEquals(destination, Tensor<float>.MatMul(x, b, destination, options)), "Destination identity");
        Equal(reference, destination.Buffer.Span, name + " overwrite destination");
        Require(Hash(x.Buffer.Span) == input && Hash(b.Buffer.Span) == weight, name + " immutable public inputs");
        if (map is not null && packedBefore is not null)
            Require(map.All(p => Hash(p.Value.Packed.Buffer.Span) == packedBefore[p.Key]), name + " immutable prepared inputs");
        Require(copies == 0 && scratches == 0, name + " no input copy or scratch");
    }

    [Fact]
    public void DimensionsRowsAndBatchesRetainBitsAndOwnership()
    {
        var normal = TensorExecutionOptions.Auto;
        foreach (int k in new[] { 8191, 8192, 8193, 8198, 8200, 8223, 8224 })
        {
            const int n = 3;
            float[] weights = Values(n * k, 13);
            var b = new DenseTensor<float>(weights, new[] { n, k }); var map = Mapping(b, weights);
            foreach (int rows in new[] { 1, 2, 3, 5 })
            {
                var x = new DenseTensor<float>(Values(rows * n, 7), new[] { rows, n });
                Public($"2d-{rows}-{k}", x, b, map, normal);
            }
            var batch = new DenseTensor<float>(Values(2 * n, 19), new[] { 2, 1, n });
            Public($"batch-{k}", batch, b, map, normal);
        }
    }

    [Fact]
    public void OptionsMissingAndReplacedWeightsRetainBitsAndOwnership()
    {
        var normal = TensorExecutionOptions.Auto;
        int index = 0;
        foreach (var selected in new[] { TensorExecutionOptions.Scalar, TensorExecutionOptions.Simd, normal })
        {
            const int n = 7, k = 8198;
            var x = new DenseTensor<float>(Values(n, 17), new[] { 1, n });
            float[] weights = Values(n * k, 11); var b = new DenseTensor<float>(weights, new[] { n, k });
            var map = Mapping(b, weights);
            string label = $"{selected.UseSimd}-{selected.UseIntrinsics}";
            // Auto equals another explicit mode on hardware-disabled hosts; name by index below.
            int id = index++;
            Public($"options-{id}-{label}", x, b, map, selected);
            Public($"missing-{id}-{label}", x, b, null, selected);
            var replacement = new DenseTensor<float>(Values(n * k, 83), new[] { n, k });
            Public($"stale-{id}-{label}", x, replacement, map, selected);
        }
    }

    [SkippableFact]
    public void RawBoundariesRetainArithmeticGuardsAndZeroAllocation()
    {
        Skip.If(!Fma.IsSupported, "FMA required");
        PackedKernel kernel = PreparedSingleRowKernel.Multiply;
        foreach (int n in new[] { 0, 1, 3, 17 })
        foreach (int k in new[] { 0, 1, 7, 8, 15, 16, 23, 24, 31, 32, 33, 63, 64, 65, 8192, 8198, 8223 })
            Raw($"boundary-{n}-{k}", n, k, Values(n, 31), Values(n * k, 43), Values(k, 61), kernel);
    }

    [SkippableFact]
    public void RawExceptionalValuesRetainNanPayloadsAndOwnedInputs()
    {
        Skip.If(!Fma.IsSupported, "FMA required");
        PackedKernel kernel = PreparedSingleRowKernel.Multiply;
        float[] special = [float.NegativeInfinity, float.PositiveInfinity, float.MaxValue, -float.MaxValue,
            float.Epsilon, -float.Epsilon, -0f, 0f, BitConverter.Int32BitsToSingle(0x7fc12345),
            BitConverter.Int32BitsToSingle(unchecked((int)0xffc54321)), 1f, -1f];
        for (int shift = 0; shift < special.Length; shift++)
            Raw($"special-{shift}", 3, 71,
                Enumerable.Range(0, 3).Select(i => special[(i + shift) % special.Length]).ToArray(),
                Enumerable.Range(0, 3 * 71).Select(i => special[(i + shift + 4) % special.Length]).ToArray(),
                Enumerable.Range(0, 71).Select(i => special[(i + shift + 7) % special.Length]).ToArray(), kernel);
    }
}
