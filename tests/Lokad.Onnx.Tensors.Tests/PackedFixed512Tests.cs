using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Tensors.Tests;

public unsafe class PackedFixed512Tests
{
    delegate bool Kernel(int rows, int reduction, int columns, float* input, float* packed, float* output);
    static Kernel Resolve(string name) => typeof(MathOps)
        .GetMethod(name, BindingFlags.Static | BindingFlags.NonPublic)!.CreateDelegate<Kernel>();

    [SkippableTheory]
    [InlineData(512)]
    [InlineData(2048)]
    [InlineData(3072)]
    [InlineData(4096)]
    [InlineData(12288)]
    public void PreservesPackedAccumulationAndUnalignedGuards(int columns)
    {
        Skip.IfNot(Avx512F.IsSupported && Fma.IsSupported, "Requires AVX-512/FMA");
        Check(columns, exceptional: false, reduction: 512);
    }

    [SkippableFact]
    public void PreservesExceptionalValueBits()
    {
        Skip.IfNot(Avx512F.IsSupported && Fma.IsSupported, "Requires AVX-512/FMA");
        Check(512, exceptional: true, reduction: 512);
    }

    [SkippableTheory]
    [InlineData(1024, false)]
    [InlineData(1024, true)]
    [InlineData(2048, false)]
    [InlineData(2048, true)]
    public void FixedWideReductionPreservesBits(int reduction, bool exceptional)
    {
        Skip.IfNot(Avx512F.IsSupported && Fma.IsSupported, "Requires AVX-512/FMA");
        Check(512, exceptional, reduction);
    }

    [Theory]
    [InlineData(32, 512, 512)]
    [InlineData(128, 511, 512)]
    [InlineData(128, 512, 768)]
    [InlineData(128, 512, 256)]
    [InlineData(256, 512, 512)]
    [InlineData(128, 1024, 768)]
    [InlineData(128, 2048, 1024)]
    [InlineData(128, 768, 512)]
    public void DeclinesUnsupportedShapesWithoutWriting(int rows, int reduction, int columns)
    {
        var input = new float[rows * reduction];
        var packed = new float[reduction * columns];
        var output = Enumerable.Repeat(-12345.5f, rows * columns + 10).ToArray();
        fixed (float* a = input, b = packed, c = output)
        {
            Assert.False(Resolve("TryPackedAvx512Fixed512Rows")(rows, reduction, columns, a, b, c + 5));
            Assert.False(Resolve("TryPackedAvx512FixedWideRows")(rows, reduction, columns, a, b, c + 5));
        }
        Assert.All(output, value => Assert.Equal(-12345.5f, value));
    }

    static void Check(int columns, bool exceptional, int reduction)
    {
        const int rows = 128, offset = 5;
        const float guard = -12345.5f;
        var random = new Random(6713);
        float[] special = { 0f, -0f, 1f, -1f, float.PositiveInfinity, float.NegativeInfinity,
            float.NaN, BitConverter.Int32BitsToSingle(0x7fc12345) };
        var input = Enumerable.Repeat(guard, rows * reduction + 2 * offset).ToArray();
        var weights = new float[reduction * columns];
        for (int i = 0; i < rows * reduction; i++)
            input[offset + i] = exceptional ? special[i % special.Length] : random.NextSingle() - .5f;
        for (int i = 0; i < weights.Length; i++)
            weights[i] = exceptional ? special[(i / 7) % special.Length] : random.NextSingle() - .5f;
        var originalInput = input.ToArray();
        var originalWeights = weights.ToArray();
        var packed = Enumerable.Repeat(guard, weights.Length + 2 * offset).ToArray();
        var expected = Enumerable.Repeat(guard, rows * columns + 2 * offset).ToArray();
        for (int i = 0; i < rows * columns; i++) expected[offset + i] = -.25f;
        var actual = expected.ToArray();
        fixed (float* a = input, b = weights, p = packed, e = expected, c = actual)
        {
            MathOps.PackPanelsB(reduction, columns, b, p + offset);
            var packedBefore = packed.ToArray();
            Assert.True(Resolve("TryPackedAvx512Rows")(rows, reduction, columns, a + offset, p + offset, e + offset));
            Assert.True(Resolve(reduction == 512 ? "TryPackedAvx512Fixed512Rows" : "TryPackedAvx512FixedWideRows")(rows, reduction, columns, a + offset, p + offset, c + offset));
            EqualBits(packedBefore, packed);
        }
        EqualBits(expected, actual);
        EqualBits(originalInput, input);
        EqualBits(originalWeights, weights);
        Assert.All(actual.Take(offset).Concat(actual.Skip(actual.Length - offset)), value => Assert.Equal(guard, value));
    }

    static void EqualBits(float[] expected, float[] actual) =>
        Assert.True(MemoryMarshal.AsBytes(expected.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(actual.AsSpan())));
}
