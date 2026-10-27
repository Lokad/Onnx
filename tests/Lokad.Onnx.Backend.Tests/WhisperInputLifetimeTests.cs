using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json.Nodes;

namespace Lokad.Onnx.Backend.Tests;

public class WhisperInputLifetimeTests
{
    [Theory]
    [InlineData("eos")] [InlineData("past")] [InlineData("malformed")] [InlineData("cancel")]
    public void CompletedWindowReleasesInputsAndCanReuseContexts(string scenario)
    {
        using var fixture = new Fixture(scenario is "past" or "cancel");
        if (scenario == "malformed") fixture.SetLogits(true);
        fixture.CancelFirst = scenario == "cancel";
        var first = Run(fixture);
        if (scenario == "malformed") Assert.IsType<InvalidDataException>(first.Error);
        else if (scenario == "cancel") Assert.IsAssignableFrom<OperationCanceledException>(first.Error);
        else CheckResult(first.Result!, fixture.TwoSteps);
        Assert.Equal(2, first.Arrays.Length);
        AssertCleanAndCollectible(fixture, first.Arrays);

        fixture.CancelFirst = false;
        fixture.SetLogits(false);
        var retry = Run(fixture);
        Assert.Null(retry.Error);
        CheckResult(retry.Result!, fixture.TwoSteps);
        AssertCleanAndCollectible(fixture, retry.Arrays);
        if (first.Result is not null) CheckResult(first.Result, fixture.TwoSteps);
        GC.KeepAlive(fixture.Owner);
    }

    [Fact]
    public void PublicResetPreservesInputsAndReleasedBuffers()
    {
        var graph = Model.Load(new OnnxModel
        {
            Opset = { [""] = 18 }, Inputs = { Desc("x", 2) }, Outputs = { Desc("y", 2) },
            Nodes = { new OnnxNode { OpType = "Identity", Inputs = ["x"], Outputs = ["y"] } }
        }).CreateExecution(null, 4096);
        var cache = graph.ReleasedBuffers!;
        Seed(cache);
        var input = DenseTensor<float>.OfValues([1, 2]);
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true));
        var output = graph.Outputs["y"];
        graph.Reset();
        Assert.Same(input, graph.Inputs["x"]);
        Assert.Same(cache, graph.ReleasedBuffers);
        Assert.True(cache.Bytes >= 64);
        Assert.Equal(new float[] { 1, 2 }, ((Tensor<float>)output).ToArray());
    }

    [Fact]
    public void PublicResetReleasesAccountingReferencesToDiscardedOutputs()
    {
        var graph = Model.Load(new OnnxModel
        {
            Opset = { [""] = 18 }, Inputs = { Desc("x", 16) }, Outputs = { Desc("y", 16) },
            Nodes = { new OnnxNode { OpType = "Mul", Inputs = ["x", "x"], Outputs = ["square"] },
                new OnnxNode { OpType = "Add", Inputs = ["square", "x"], Outputs = ["y"] } }
        }).CreateExecution(ExecutionOptions.Memory, 4096);
        var output = ExecuteForWeakOutput(graph);
        long peak = graph.LastPeakLiveBytes;
        int allocations = graph.LastPoolAllocatedNew;
        var elapsed = graph.LastRunTime;
        graph.Reset();
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        Assert.Equal(0, Alive(new[] { output }));
        Assert.Equal(peak, graph.LastPeakLiveBytes);
        Assert.Equal(allocations, graph.LastPoolAllocatedNew);
        Assert.Equal(elapsed, graph.LastRunTime);
        Assert.Equal(Enumerable.Repeat(1f, 16), ((Tensor<float>)graph.Inputs["x"]).ToArray());
        Assert.True(graph.ReleasedBuffers!.Bytes > 0);
        GC.KeepAlive(graph);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    static WeakReference<float[]> ExecuteForWeakOutput(GraphExecution graph)
    {
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = DenseTensor<float>.Ones(16) }, true));
        var output = (Tensor<float>)graph.Outputs["y"];
        Assert.True(MemoryMarshal.TryGetArray((ReadOnlyMemory<float>)output.Storage, out var memory));
        return new WeakReference<float[]>(memory.Array!);
    }

    static void CheckResult(WhisperTranscription result, bool twoSteps)
    {
        Assert.Equal(WhisperStopReason.EndToken, result.StopReason);
        Assert.Equal(twoSteps ? new[] { 32, 50257 } : new[] { 50257 }, result.TokenIds);
        Assert.Equal(twoSteps ? "A" : "", result.Text);
        Assert.False(result.SkippedAsNoSpeech);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    static (WhisperTranscription? Result, Exception? Error, WeakReference<float[]>[] Arrays) Run(Fixture fixture)
    {
        fixture.Captured = new WeakReference<float[]>?[2];
        try
        {
            var result = fixture.Owner.Transcribe(new float[] { .5f }, 16000,
                new WhisperTranscriptionOptions("en", 2, null, null), fixture.CancelFirst ? fixture.Cancellation.Token : CancellationToken.None);
            return (result, null, fixture.Captured.OfType<WeakReference<float[]>>().ToArray());
        }
        catch (Exception ex) { return (null, ex, fixture.Captured.OfType<WeakReference<float[]>>().ToArray()); }
    }

    static void AssertCleanAndCollectible(Fixture fixture, WeakReference<float[]>[] arrays)
    {
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        Assert.True(Alive(arrays) == 0, "Retained array lengths: " + AliveLengths(arrays));
        foreach (var context in fixture.Contexts)
        {
            Assert.Equal(fixture.Keys[context], context.Inputs.Keys.ToArray());
            Assert.All(context.Inputs.Values, value => Assert.Null(value));
            Assert.All(context.Outputs.Values, value => Assert.Null(value));
            Assert.Same(fixture.Caches[context], context.ReleasedBuffers);
            Assert.True(context.ReleasedBuffers!.Bytes >= 64);
        }
        GC.KeepAlive(fixture.Owner);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    static int Alive(WeakReference<float[]>[] arrays) => arrays.Count(array => array.TryGetTarget(out _));

    [MethodImpl(MethodImplOptions.NoInlining)]
    static string AliveLengths(WeakReference<float[]>[] arrays) => string.Join(",", arrays.Select(array => array.TryGetTarget(out var value) ? value.Length : 0));

    static OnnxValueInfo Desc(string name, params int[] shape) => new() { Name = name, ElementType = TensorElementType.Float, Dims = shape };
    static OnnxTensor Constant(string name, int[] shape, float[] values) => new() { Name = name, ElementType = TensorElementType.Float, Dims = shape, Data = values };
    static void Seed(ReleasedBufferCache cache)
    {
        var pool = new TensorBufferPool(cache);
        pool.Return(pool.Rent<float>(16));
        pool.RetainReleasedBuffers();
    }

    // A runtime-dependent multiply reads this test scalar while inputs are
    // bound. Capture only weak references; avoid global logging/production hooks.
    sealed class Tap(Action observe) : Tensor<float>(Array.Empty<int>(), false)
    {
        public override float GetValue(int index) { observe(); return 0; }
        public override void SetValue(int index, float value) => throw new NotSupportedException();
        public override Tensor<float> Clone() => DenseTensor<float>.Scalar(GetValue(0));
        public override Tensor<T> CloneEmpty<T>(ReadOnlySpan<int> shape) => new DenseTensor<T>(shape);
        public override Tensor<float> Reshape(ReadOnlySpan<int> shape) => Clone().Reshape(shape);
    }

    sealed class Fixture : IDisposable
    {
        internal readonly WhisperTranscriber Owner;
        internal readonly bool TwoSteps;
        internal readonly GraphExecution[] Contexts;
        internal readonly Dictionary<GraphExecution, string[]> Keys;
        internal readonly Dictionary<GraphExecution, ReleasedBufferCache> Caches;
        internal readonly CancellationTokenSource Cancellation = new();
        internal bool CancelFirst;
        internal WeakReference<float[]>?[] Captured = new WeakReference<float[]>?[2];

        internal Fixture(bool twoSteps)
        {
            TwoSteps = twoSteps;
            var metadata = JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "whisper-text.json")))!;
            var assembly = typeof(WhisperTranscriber).Assembly;
            using var tokens = new MemoryStream(Encoding.UTF8.GetBytes(metadata["tokenizer"]!.ToJsonString()));
            var tokenizer = Activator.CreateInstance(assembly.GetType("Lokad.Onnx.WhisperTokenizer", true)!,
                BindingFlags.Instance | BindingFlags.NonPublic, null, new object[] { tokens }, null)!;
            if (!twoSteps) metadata["generation"]!["begin_suppress_tokens"] = new JsonArray();
            using var config = new MemoryStream(Encoding.UTF8.GetBytes(metadata["generation"]!.ToJsonString()));
            var generation = Activator.CreateInstance(assembly.GetType("Lokad.Onnx.WhisperGeneration", true)!,
                BindingFlags.Instance | BindingFlags.NonPublic, null, new[] { (object)config, tokenizer }, null)!;

            var encoder = new OnnxModel
            {
                Opset = { [""] = 18 }, Inputs = { Desc("input_features", 1, 128, 3000) }, Outputs = { Desc("last_hidden_state", 1, 1500, 1280) },
                Initializers = { Constant("base", [1, 1500, 1280], new float[1500 * 1280]), Constant("tap", [], [0]),
                    new OnnxTensor { Name = "axes", ElementType = TensorElementType.Int64, Dims = [3], Data = new long[] { 0, 1, 2 } } },
                Nodes = { new OnnxNode { OpType = "ReduceMean", Inputs = ["input_features", "axes"], Outputs = ["mean"], Attributes = new() { ["keepdims"] = 0L } },
                    new OnnxNode { OpType = "Mul", Inputs = ["mean", "tap"], Outputs = ["zero"] },
                    new OnnxNode { OpType = "Add", Inputs = ["base", "zero"], Outputs = ["last_hidden_state"] } }
            };
            var first = Decoder(4);
            first.Inputs.Add(Desc("encoder_hidden_states", 1, 1500, 1280));
            var past = Decoder(1);
            if (twoSteps)
            {
                first.Initializers.Add(new OnnxTensor { Name = "cross-shape", ElementType = TensorElementType.Int64, Dims = [4], Data = new long[] { 1, 20, 1500, 64 } });
                first.Initializers.Add(Constant("self", [1, 20, 4, 64], new float[20 * 4 * 64]));
                for (int layer = 0; layer < 4; layer++)
                foreach (string attention in new[] { "decoder", "encoder" })
                foreach (string kind in new[] { "key", "value" })
                {
                    string name = $"{layer}.{attention}.{kind}";
                    int[] shape = [1, 20, attention == "decoder" ? 4 : 1500, 64];
                    first.Outputs.Add(Desc("present." + name, shape));
                    first.Nodes.Add(new OnnxNode { OpType = attention == "decoder" ? "Identity" : "Reshape",
                        Inputs = attention == "decoder" ? ["self"] : ["encoder_hidden_states", "cross-shape"], Outputs = ["present." + name] });
                    past.Inputs.Add(Desc("past_key_values." + name, shape));
                }
            }
            Contexts = new[] { Model.Load(encoder).CreateExecution(ExecutionOptions.Memory, 4096),
                Model.Load(first).CreateExecution(ExecutionOptions.Memory, 4096), Model.Load(past).CreateExecution(ExecutionOptions.Memory, 4096) };
            Contexts[0].Initializers["tap"] = new Tap(() => Capture(0, "input_features", 0));
            Contexts[1].Initializers["tap"] = new Tap(() => { Capture(1, "encoder_hidden_states", 1); if (CancelFirst) Cancellation.Cancel(); });
            Keys = Contexts.ToDictionary(context => context, context => context.Inputs.Keys.ToArray());
            Caches = Contexts.ToDictionary(context => context, context => context.ReleasedBuffers!);
            foreach (var cache in Caches.Values) Seed(cache);
            Owner = (WhisperTranscriber)RuntimeHelpers.GetUninitializedObject(typeof(WhisperTranscriber));
            Set("gate", new object()); Set("generation", generation);
            Set("encodingExecution", Contexts[0]); Set("firstExecution", Contexts[1]); Set("pastExecution", Contexts[2]);
            SetLogits(false);

            void Set(string name, object value) => typeof(WhisperTranscriber).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(Owner, value);
        }

        static OnnxModel Decoder(int prefix) => new()
        {
            Opset = { [""] = 18 }, Inputs = { new OnnxValueInfo { Name = "input_ids", ElementType = TensorElementType.Int64, Dims = [1, prefix] } },
            Outputs = { Desc("logits", 1, prefix, 51866) }, Initializers = { Constant("tap", [], [0]),
                Constant("base", [1, prefix, 51866], new float[prefix * 51866]) },
            Nodes = { new OnnxNode { OpType = "Cast", Inputs = ["input_ids"], Outputs = ["ids"], Attributes = new() { ["to"] = (long)TensorElementType.Float } },
                new OnnxNode { OpType = "ReduceMean", Inputs = ["ids"], Outputs = ["mean"], Attributes = new() { ["keepdims"] = 0L } },
                new OnnxNode { OpType = "Mul", Inputs = ["mean", "tap"], Outputs = ["zero"] },
                new OnnxNode { OpType = "Add", Inputs = ["base", "zero"], Outputs = ["logits"] } }
        };

        internal void SetLogits(bool malformed)
        {
            for (int i = 1; i < 3; i++)
            {
                int prefix = i == 1 && !malformed ? 4 : 1;
                var values = Enumerable.Repeat(-100f, prefix * 51866).ToArray();
                for (int row = 0; row < prefix; row++) values[row * 51866 + (i == 1 && TwoSteps ? 32 : 50257)] = 0;
                Contexts[i].Initializers["base"] = new DenseTensor<float>(values, [1, prefix, 51866]);
                Contexts[i].OutputDescs[0].Dims = [1, prefix, 51866];
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        void Capture(int context, string name, int slot)
        {
            var tensor = (Tensor<float>)Contexts[context].Inputs[name];
            Assert.True(MemoryMarshal.TryGetArray((ReadOnlyMemory<float>)tensor.Storage, out var memory));
            Captured[slot] = new WeakReference<float[]>(memory.Array!);
        }
        public void Dispose() => Cancellation.Dispose();
    }
}
