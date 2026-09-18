using System.Text.Json;

namespace Lokad.Onnx.Backend.Tests;

public class LstmReferenceTests
{
    static JsonDocument Reference() => JsonDocument.Parse(File.ReadAllText(
        Path.Combine(AppContext.BaseDirectory, "fixtures", "lstm-ort.json")));

    public static IEnumerable<object[]> Cases()
    {
        using var reference = Reference();
        foreach (var item in reference.RootElement.GetProperty("cases").EnumerateArray())
            for (int mode = 0; mode < 3; mode++)
                yield return new object[] { item.GetProperty("name").GetString()!, mode };
    }

    static DenseTensor<float> Tensor(JsonElement value, int mode)
    {
        int[] shape = value.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        float[] data = value.GetProperty("values").EnumerateArray().Select(x => x.GetSingle()).ToArray();
        // Exercise sliced backing memory and column-major strides, including weights/states.
        var memory = Enumerable.Repeat(float.NaN, data.Length + 4).ToArray().AsMemory(2, data.Length);
        var tensor = new DenseTensor<float>(mode == 0 ? data.AsMemory() : memory, shape, mode == 2);
        if (mode != 0)
        {
            var indices = new int[shape.Length];
            for (int i = 0; i < data.Length; i++)
            {
                int flat = i;
                for (int d = shape.Length - 1; d >= 0; d--) { indices[d] = flat % shape[d]; flat /= shape[d]; }
                tensor[indices] = data[i];
            }
        }
        return tensor;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void MultipleBatchesDirectionsActivationsAndStorageMatchOrt(string name, int mode)
    {
        using var reference = Reference();
        var item = reference.RootElement.GetProperty("cases").EnumerateArray().Single(x => x.GetProperty("name").GetString() == name);
        var inputs = item.GetProperty("inputs");
        var tensors = inputs.EnumerateObject().Where(p => p.Name != "sequence_lens")
            .ToDictionary(p => p.Name, p => Tensor(p.Value, mode));
        var before = tensors.ToDictionary(p => p.Key, p => p.Value.ToArray());
        var lens = inputs.GetProperty("sequence_lens").GetProperty("values").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var lengths = DenseTensor<int>.OfValues(lens);
        var attrs = item.GetProperty("attributes");
        float[]? Floats(string key) => attrs.TryGetProperty(key, out var value) ? value.EnumerateArray().Select(x => x.GetSingle()).ToArray() : null;
        string[]? acts = attrs.TryGetProperty("activations", out var a) ? a.EnumerateArray().Select(x => x.GetString()!).ToArray() : null;
        float? clip = attrs.TryGetProperty("clip", out var c) ? c.GetSingle() : null;
        bool coupled = attrs.TryGetProperty("input_forget", out var f) && f.GetInt32() != 0;
        var options = mode == 0 ? ExecutionOptions.Scalar : mode == 1 ? ExecutionOptions.Simd : ExecutionOptions.Intrinsics;
        var pool = new TensorBufferPool();
        var retained = new List<(Tensor<float> Tensor, float[] Values)>();
        var expected = item.GetProperty("outputs").EnumerateArray().ToArray();
        foreach (var output in expected)
        {
            int count = output.GetProperty("values").GetArrayLength();
            pool.Return(Enumerable.Repeat(float.NaN, count).ToArray());
        }
        for (int outputs = 1; outputs <= 3; outputs++)
        {
            var result = CPUExecutionProvider.Lstm(tensors["X"], tensors["W"], tensors["R"], tensors["B"], lengths,
                tensors["initial_h"], tensors["initial_c"], tensors["P"], attrs.GetProperty("direction").GetString(),
                acts, Floats("activation_alpha"), Floats("activation_beta"), clip, 3, coupled, 0, outputs, options, pool);
            Assert.True(result.Status == OpStatus.Success, result.Message);
            Assert.Equal(outputs, result.Outputs.Length);
            for (int j = 0; j < outputs; j++)
            {
                var tensor = (Tensor<float>)result.Outputs[j];
                Assert.Equal(expected[j].GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()), tensor.Dimensions.ToArray());
                float[] values = tensor.ToArray();
                float[] oracle = expected[j].GetProperty("values").EnumerateArray().Select(x => x.GetSingle()).ToArray();
                Assert.Equal(oracle.Length, values.Length);
                for (int k = 0; k < oracle.Length; k++)
                    Assert.True(float.IsFinite(values[k]) && Math.Abs(values[k] - oracle[k]) <= 1e-5f * Math.Max(1f, Math.Abs(oracle[k])),
                        $"{name} output {j}[{k}]: {values[k]} vs {oracle[k]}");
                retained.Add((tensor, values));
            }
            foreach (var old in retained) Assert.Equal(old.Values, old.Tensor.ToArray());
            foreach (var pair in tensors) Assert.Equal(before[pair.Key], pair.Value.ToArray());
            Assert.Equal(lens, lengths.ToArray());
        }
        Assert.True(pool.Reused >= 3);
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(-1f)]
    [InlineData(float.NaN)]
    public void InvalidClipReturnsFailure(float clip)
    {
        var result = CPUExecutionProvider.Lstm(new DenseTensor<float>(new[] { 1, 1, 1 }),
            new DenseTensor<float>(new[] { 1, 4, 1 }), new DenseTensor<float>(new[] { 1, 4, 1 }),
            null, null, null, null, null, null, null, null, null, clip, 1, false, 0, 3, null, null);
        Assert.Equal(OpStatus.Failure, result.Status);
        Assert.Contains("clip", result.Message);
    }

    [Fact]
    public void RankOneBiasIsRejected()
    {
        var result = CPUExecutionProvider.Lstm(new DenseTensor<float>(new[] { 1, 1, 1 }),
            new DenseTensor<float>(new[] { 1, 4, 1 }), new DenseTensor<float>(new[] { 1, 4, 1 }),
            new DenseTensor<float>(8), null, null, null, null, null, null, null, null, null, 1, false, 0, 3, null, null);
        Assert.Equal(OpStatus.Failure, result.Status);
        Assert.Contains("B", result.Message);
    }

    [Theory]
    [InlineData(1L, true)]
    [InlineData(4294967297L, false)]
    [InlineData(-4294967295L, false)]
    public void Int64SequenceLengthsAreValidatedBeforeNarrowing(long length, bool success)
    {
        var result = CPUExecutionProvider.Lstm(new DenseTensor<float>(new[] { 1, 1, 1 }),
            new DenseTensor<float>(new[] { 1, 4, 1 }), new DenseTensor<float>(new[] { 1, 4, 1 }),
            null, DenseTensor<long>.OfValues(new[] { length }), null, null, null,
            null, null, null, null, null, 1, false, 0, 3, null, null);
        Assert.Equal(success ? OpStatus.Success : OpStatus.Failure, result.Status);
    }

    [Theory]
    [InlineData(0, 2)]
    [InlineData(2, 0)]
    public void EmptySequenceOrBatchProducesInitializedEmptyOutputs(int sequence, int batch)
    {
        var result = CPUExecutionProvider.Lstm(new DenseTensor<float>(new[] { sequence, batch, 1 }),
            new DenseTensor<float>(new[] { 1, 4, 1 }), new DenseTensor<float>(new[] { 1, 4, 1 }),
            null, null, null, new DenseTensor<float>(Enumerable.Repeat(10f, batch).ToArray(), new[] { 1, batch, 1 }), null,
            null, null, null, null, null, 1, false, 0, 3, null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Empty(((Tensor<float>)result.Outputs[0]).ToArray());
        Assert.Equal(new float[batch], ((Tensor<float>)result.Outputs[1]).ToArray());
        Assert.Equal(new float[batch], ((Tensor<float>)result.Outputs[2]).ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(int.MaxValue)]
    public void InvalidHiddenSizeIsRejectedBeforeArithmetic(int hidden)
    {
        var result = CPUExecutionProvider.Lstm(new DenseTensor<float>(new[] { 1, 1, 1 }),
            new DenseTensor<float>(new[] { 1, 4, 1 }), new DenseTensor<float>(new[] { 1, 4, 1 }),
            null, null, null, null, null, null, null, null, null, null, hidden, false, 0, 3, null, null);
        Assert.Equal(OpStatus.Failure, result.Status);
        Assert.Contains("hidden_size", result.Message);
    }

    [Theory]
    [InlineData(0, false)]
    [InlineData(1, true)]
    [InlineData(2, false)]
    [InlineData(3, true)]
    public void ImportedGraphHonorsOptionalOutputSlotsAndRetainedStates(int selection, bool context)
    {
        using var reference = Reference();
        var item = reference.RootElement.GetProperty("cases").EnumerateArray().Single(x => x.GetProperty("name").GetString() == "partial_lists_use_defaults");
        var inputDefs = item.GetProperty("inputs");
        var expected = item.GetProperty("outputs").EnumerateArray().ToArray();
        string[] outputs = selection == 0 ? new[] { "y" } : selection == 1 ? new[] { "", "yh" }
            : selection == 2 ? new[] { "", "", "yc" } : new[] { "y", "yh", "yc" };
        var model = new OnnxModel { Name = "lstm-optional-states" };
        model.Opset[""] = 17;
        model.Inputs.Add(new OnnxValueInfo { Name = "X", ElementType = TensorElementType.Float, Dims = new[] { 4, 3, 2 } });
        foreach (var input in inputDefs.EnumerateObject().Where(p => p.Name != "X"))
        {
            bool integer = input.Name == "sequence_lens";
            Array values = integer ? input.Value.GetProperty("values").EnumerateArray().Select(x => x.GetInt32()).ToArray()
                : input.Value.GetProperty("values").EnumerateArray().Select(x => x.GetSingle()).ToArray();
            model.Initializers.Add(new OnnxTensor { Name = input.Name, ElementType = integer ? TensorElementType.Int32 : TensorElementType.Float,
                Dims = input.Value.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray(), Data = values });
        }
        for (int i = 0; i < outputs.Length; i++)
            if (outputs[i].Length != 0) model.Outputs.Add(new OnnxValueInfo { Name = outputs[i], ElementType = TensorElementType.Float,
                Dims = expected[i].GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray() });
        model.Nodes.Add(new OnnxNode { Name = "recurrent", OpType = "LSTM", Inputs = inputDefs.EnumerateObject().Select(p => p.Name).ToArray(), Outputs = outputs,
            Attributes = new Dictionary<string, object> { ["hidden_size"] = 3L, ["direction"] = "bidirectional",
                ["activations"] = new[] { "Sigmoid", "LeakyRelu", "Tanh", "HardSigmoid", "Tanh", "Tanh" }, ["activation_alpha"] = new[] { .3f } } });
        var plan = Model.Load(model)!;
        var graph = context ? plan.CreateExecution(null) : plan;
        var retained = new List<(Tensor<float> Tensor, float[] Values)>();
        for (int run = 0; run < 3; run++)
        {
            graph.Reset();
            var input = Tensor(inputDefs.GetProperty("X"), 0);
            for (int i = 0; i < input.Length; i++) input.SetValue(i, input.GetValue(i) + run * .1f);
            var before = input.ToArray();
            Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["X"] = input }, true), graph.LastErrorMessage);
            Assert.Equal(before, input.ToArray());
            Assert.False(graph.Outputs.ContainsKey(""));
            Assert.False(graph.IntermediateOutputs.ContainsKey(""));
            foreach (var old in retained) Assert.Equal(old.Values, old.Tensor.ToArray());
            for (int i = 0; i < outputs.Length; i++)
            {
                if (outputs[i].Length == 0) continue;
                var tensor = (Tensor<float>)graph.Outputs[outputs[i]];
                var values = tensor.ToArray();
                if (run == 0)
                {
                    var oracle = expected[i].GetProperty("values").EnumerateArray().Select(x => x.GetSingle()).ToArray();
                    for (int j = 0; j < values.Length; j++) Assert.InRange(Math.Abs(values[j] - oracle[j]), 0f, 1e-5f);
                }
                retained.Add((tensor, values));
            }
            graph.Reset();
            Assert.False(graph.Execute(new Dictionary<string, ITensor>(), false));
            foreach (var old in retained) Assert.Equal(old.Values, old.Tensor.ToArray());
        }
    }
}
