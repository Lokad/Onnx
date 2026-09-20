namespace Lokad.Onnx.Backend.Tests;

using System.Reflection;
using System.Runtime.InteropServices;

public class WhisperDecoderWeightsTests
{
    static long Share(ComputationalGraph first, ComputationalGraph past) => (long)typeof(WhisperTranscriber).Assembly
        .GetType("Lokad.Onnx.WhisperDecoderWeights", true)!
        .GetMethod("Share", BindingFlags.Static | BindingFlags.NonPublic)!.Invoke(null, new object[] { first, past })!;

    static DenseTensor<float> Weight(float[]? values = null, int[]? shape = null) =>
        new DenseTensor<float>(values ?? Enumerable.Range(0, 1024).Select(i => (float)i).ToArray(), shape ?? new[] { 32, 32 });

    static float[] Storage(ITensor tensor)
    {
        Assert.True(MemoryMarshal.TryGetArray(((DenseTensor<float>)tensor).Buffer, out ArraySegment<float> array));
        Assert.Equal(0, array.Offset);return array.Array!;
    }

    [Fact]
    public void ExactWeightsShareOnlyTheirBackingStorageAndRetainSeparateMetadata()
    {
        var first = new ComputationalGraph();var past = new ComputationalGraph();
        var a = Weight();var b = Weight();a.Name = "first-name";b.Name = "past-name";
        first.Initializers["first-key"] = a;past.Initializers["past-key"] = b;
        var original = b.ToArray();
        Assert.Equal(4096, Share(first, past));
        var shared = past.Initializers["past-key"];
        Assert.NotSame(a, shared);Assert.NotSame(b, shared);
        Assert.Same(Storage(a), Storage(shared));Assert.NotSame(Storage(b), Storage(shared));
        Assert.Equal("first-name", a.Name);Assert.Equal("past-name", shared.Name);
        Assert.Equal(new[] { 32, 32 }, shared.Dims);Assert.Equal(original, ((Tensor<float>)shared).ToArray());
        Assert.Equal(original, b.ToArray());Assert.Equal(0, Share(first, past));
    }

    [Theory]
    [InlineData(0, unchecked((int)0x80000000))]
    [InlineData(unchecked((int)0x7fc00001), unchecked((int)0x7fc00002))]
    public void DifferentSignedZeroOrNanBitsNeverShare(int aBits, int bBits)
    {
        var first = new ComputationalGraph();var past = new ComputationalGraph();
        var a = new float[1024];var b = new float[1024];a[0] = BitConverter.Int32BitsToSingle(aBits);b[0] = BitConverter.Int32BitsToSingle(bBits);
        first.Initializers["a"] = Weight(a);var old = past.Initializers["b"] = Weight(b);
        Assert.Equal(0, Share(first, past));Assert.Same(old, past.Initializers["b"]);
        Assert.Equal(aBits, BitConverter.SingleToInt32Bits(a[0]));Assert.Equal(bBits, BitConverter.SingleToInt32Bits(b[0]));
    }

    [Theory]
    [InlineData("shape")]
    [InlineData("reversed")]
    [InlineData("slice")]
    [InlineData("small")]
    [InlineData("input")]
    [InlineData("output")]
    [InlineData("descriptor-input")]
    [InlineData("descriptor-output")]
    public void IneligibleOrObservableStorageRemainsIndependent(string mode)
    {
        var first = new ComputationalGraph();var past = new ComputationalGraph();
        DenseTensor<float> a = Weight(), b = Weight();
        if (mode == "shape") b = Weight(shape: new[] { 16, 64 });
        if (mode == "reversed") b = new DenseTensor<float>(b.Buffer, new[] { 32, 32 }, true);
        if (mode == "slice") { var backing = new float[1025];a.Buffer.Span.CopyTo(backing.AsSpan(1));b = new DenseTensor<float>(backing.AsMemory(1), new[] { 32, 32 }); }
        if (mode == "small") { a = Weight(new float[1023], new[] { 1023 });b = Weight(new float[1023], new[] { 1023 }); }
        if (mode == "input") past.Inputs["w"] = null;
        if (mode == "output") past.Outputs["w"] = null;
        if (mode == "descriptor-input") past.InputDescs.Add(new OnnxValueInfo { Name = "w" });
        if (mode == "descriptor-output") past.OutputDescs.Add(new OnnxValueInfo { Name = "w" });
        first.Initializers["a"] = a;past.Initializers["w"] = b;
        Assert.Equal(0, Share(first, past));Assert.Same(b, past.Initializers["w"]);
    }

    static ComputationalGraph MatMul()
    {
        var graph = new ComputationalGraph();graph.Metadata["Name"] = "shared-private-decoder-weights";var weights = new float[1024];
        for (int i = 0; i < 32; i++) weights[i * 32 + i] = 2;
        graph.Initializers["w"] = Weight(weights);graph.Inputs["x"] = null;graph.Outputs["y"] = null;
        graph.Nodes.Add(new Node { Name = "multiply", Op = OpType.MatMul, OpTypeName = "MatMul", Inputs = new[] { "x", "w" }, Outputs = new[] { "y" } });
        graph.InputDescs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 1, 32 } });
        graph.OutputDescs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { 1, 32 } });
        graph.Prepare();return graph;
    }

    [Fact]
    public void PreparedGraphsRebuildDerivedWeightsAndPreserveHeldOutputsAndInputs()
    {
        var first = MatMul();var past = MatMul();
        Assert.NotEmpty(past.PackedWeights);
        var initial = new DenseTensor<float>(Enumerable.Range(1, 32).Select(x => (float)x).ToArray(), new[] { 1, 32 });
        Assert.True(past.Execute(new Dictionary<string, ITensor> { ["x"] = initial }, true));
        var held = (Tensor<float>)past.Outputs["y"]!;var heldValues = held.ToArray();
        Assert.Equal(4096, Share(first, past));Assert.Empty(past.PackedWeights);
        var all = new List<(Tensor<float> Tensor, float[] Values)> { (held, heldValues) };
        for (int run = 0; run < 4; run++) foreach (var graph in new[] { first, past })
        {
            var values = Enumerable.Range(run * 32, 32).Select(x => (float)x).ToArray();var input = new DenseTensor<float>(values, new[] { 1, 32 });
            Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true));
            var output = (Tensor<float>)graph.Outputs["y"]!;Assert.Equal(values.Select(x => x * 2), output.ToArray());
            Assert.Equal(values, input.ToArray());all.Add((output, output.ToArray()));
            foreach (var old in all) Assert.Equal(old.Values, old.Tensor.ToArray());
        }
        Assert.NotEmpty(past.PackedWeights);Assert.Equal(Enumerable.Range(1, 32).Select(x => (float)x), initial.ToArray());
        Assert.Same(Storage(first.Initializers["w"]), Storage(past.Initializers["w"]));
    }
}
