using System.Reflection;

namespace Lokad.Onnx.Backend.Tests;

[Collection("NonParallelAllocation")]
public class CompanionOutputValidationTests
{
    static readonly Func<IReadOnlyDictionary<string, ITensor>, string, int[], Tensor<float>> Parakeet =
        typeof(ParakeetTranscriber).Assembly.GetType("Lokad.Onnx.ParakeetGeneration", true)!
            .GetMethod("RequireFloat", BindingFlags.Static | BindingFlags.NonPublic)!
            .CreateDelegate<Func<IReadOnlyDictionary<string, ITensor>, string, int[], Tensor<float>>>();

    static readonly Func<GraphExecution, string, Tensor<float>, string, int[], Tensor<float>> WeSpeaker =
        typeof(WeSpeakerEmbedder).GetMethod("Execute", BindingFlags.Static | BindingFlags.NonPublic)!
            .CreateDelegate<Func<GraphExecution, string, Tensor<float>, string, int[], Tensor<float>>>();

    sealed class Fixture : IDisposable
    {
        internal readonly Tensor<float> Input, Selected;
        internal readonly int[] Shape = { 1, 2, 3 };
        internal readonly GraphExecution Execution;
        readonly Dictionary<string, ITensor> outputs;

        internal Fixture(int layout)
        {
            Input = layout switch
            {
                1 => new DenseTensor<float>(new[] { 1, 2, 4 }, true),
                2 => DenseTensor<float>.OfShape(1, 1, 4).BroadcastDim(1, 2),
                3 => DenseTensor<float>.OfShape(1, 2, 5).Slice(SliceIndex.All, SliceIndex.All, new SliceIndex(1, 5)),
                _ => DenseTensor<float>.OfShape(1, 2, 4)
            };
            Input.Fill(1);
            Input[0, 0, 3] = Input[0, 1, 3] = float.NaN; // Padding outside the selected view.
            Selected = Input.Slice(SliceIndex.All, SliceIndex.All, new SliceIndex(0, 3));
            outputs = new Dictionary<string, ITensor> { ["y"] = Selected };
            var model = new OnnxModel
            {
                Opset = { [""] = 18 },
                Inputs = { new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = Input.Dimensions.ToArray() } },
                Outputs = { new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = Shape } },
                Initializers =
                {
                    new OnnxTensor { Name = "start", ElementType = TensorElementType.Int64, Dims = new[] { 1 }, Data = new long[] { 0 } },
                    new OnnxTensor { Name = "end", ElementType = TensorElementType.Int64, Dims = new[] { 1 }, Data = new long[] { 3 } },
                    new OnnxTensor { Name = "axis", ElementType = TensorElementType.Int64, Dims = new[] { 1 }, Data = new long[] { 2 } }
                },
                Nodes = { new OnnxNode { OpType = "Slice", Inputs = new[] { "x", "start", "end", "axis" }, Outputs = new[] { "y" } } }
            };
            Execution = Model.Load(model, 0).CreateExecution(ExecutionOptions.Memory);
        }

        internal Tensor<float> Read(bool speaker, int[]? shape = null)
        {
            Execution.Reset();
            return speaker ? WeSpeaker(Execution, "x", Input, "y", shape ?? Shape) : Parakeet(outputs, "y", shape ?? Shape);
        }

        public void Dispose() => Execution.Reset();
    }

    [Theory]
    [InlineData(false, 0)] [InlineData(true, 0)]
    [InlineData(false, 1)] [InlineData(true, 1)]
    [InlineData(false, 2)] [InlineData(true, 2)]
    [InlineData(false, 3)] [InlineData(true, 3)]
    public void ValidationReadsSelectedValuesAndRetainsStorage(bool speaker, int layout)
    {
        using var fixture = new Fixture(layout);
        float[] before = fixture.Input.Storage.ToArray();
        var result = fixture.Read(speaker);
        Assert.Same(speaker ? fixture.Execution.Outputs["y"] : fixture.Selected, result);
        Assert.Equal(fixture.Input.Storage, result.Storage);
        Assert.Equal(new[] { 1f, 1f, 1f, 1f, 1f, 1f }, result.ToArray());
        Assert.Equal(before, fixture.Input.Storage.ToArray());
        result[0, 1, 2] = 7;
        Assert.Equal(7, fixture.Input[0, 1, 2]);
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void ValidationPreservesFailuresAndAllowsRetry(bool speaker)
    {
        using var fixture = new Fixture(0);
        foreach (float bad in new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity })
        {
            fixture.Input[0, 0, 0] = bad;
            var error = Assert.Throws<InvalidDataException>(() => fixture.Read(speaker));
            Assert.Equal(speaker ? "Nonfinite WeSpeaker output: y" : "Unexpected or nonfinite Parakeet output: y", error.Message);
            fixture.Input[0, 0, 0] = 1;
            Assert.Equal(1, fixture.Read(speaker)[0, 0, 0]);
        }
        var shapeError = Assert.Throws<InvalidDataException>(() => fixture.Read(speaker, new[] { 1, 2, 4 }));
        Assert.Equal(speaker ? "Unexpected WeSpeaker output: y" : "Unexpected or nonfinite Parakeet output: y", shapeError.Message);
        Assert.Equal(1, fixture.Read(speaker)[0, 0, 0]);
        if (!speaker)
        {
            Assert.Throws<InvalidDataException>(() => Parakeet(new Dictionary<string, ITensor>(), "y", fixture.Shape));
            Assert.Throws<InvalidDataException>(() => Parakeet(new Dictionary<string, ITensor>
                { ["y"] = DenseTensor<int>.OfShape(1, 2, 3) }, "y", fixture.Shape));
        }
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public async Task ValidationAllocationDoesNotGrowWithPayload(bool speaker)
    {
        long allocated = await Task.Factory.StartNew(() =>
        {
            int[] shape = { 1, speaker ? 2560 : 1024, 256 };
            var source = new DenseTensor<float>(shape);
            source.Fill(1);
            var outputs = new Dictionary<string, ITensor> { ["y"] = source };
            var model = new OnnxModel
            {
                Inputs = { new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = shape } },
                Outputs = { new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = shape } },
                Nodes = { new OnnxNode { OpType = "Identity", Inputs = new[] { "x" }, Outputs = new[] { "y" } } }
            };
            var context = Model.Load(model, 0).CreateExecution(ExecutionOptions.Memory);
            Tensor<float> Read()
            {
                context.Reset();
                return speaker ? WeSpeaker(context, "x", source, "y", shape) : Parakeet(outputs, "y", shape);
            }
            try
            {
                for (int warm = 0; warm < 256; warm++) Read();
                long before = GC.GetAllocatedBytesForCurrentThread();
                var result = Read();
                long bytes = GC.GetAllocatedBytesForCurrentThread() - before;
                Assert.Equal(source.Storage, result.Storage);
                Assert.Equal(1, result.GetValue(0));
                return bytes;
            }
            finally { context.Reset(); }
        }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        Assert.InRange(allocated, 0, 16_384);
    }
}
