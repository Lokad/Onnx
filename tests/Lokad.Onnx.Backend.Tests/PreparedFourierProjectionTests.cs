using System.Buffers;
using System.Numerics;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

public class PreparedFourierProjectionTests
{
    static DenseTensor<float> Weights(int n)
    {
        int bins = n / 2 + 1;
        var result = new float[2 * bins * n];
        for (int k = 0; k < bins; k++)
            for (int j = 0; j < n; j++)
            {
                double phase = 2 * Math.PI * ((k * j) % n) / n;
                double window = .5 - .5 * Math.Cos(2 * Math.PI * j / n);
                result[k * n + j] = (float)(Math.Cos(phase) * window);
                result[(bins + k) * n + j] = (float)(-Math.Sin(phase) * window);
            }
        return new(result, [2 * bins, 1, n]);
    }

    static DenseTensor<float> Input(int n, int frames, int step) => new(
        Enumerable.Range(0, n + (frames - 1) * step).Select(i => (float)(Math.Sin(i * .37) * .2 + Math.Cos(i * .11) * .3)).ToArray(),
        [1, 1, n + (frames - 1) * step]);

    static float[] Reference(Tensor<float> input, Tensor<float> weight, int step, ExecutionOptions? options = null)
    {
        var result = CPUExecutionProvider.Conv(input, weight, null, null, [1], 1, [weight.Dimensions[2]], [0, 0], [step], options ?? ExecutionOptions.Scalar, null);
        Assert.Equal(OpStatus.Success, result.Status);
        return ((Tensor<float>)Assert.Single(result.Outputs)).ToArray();
    }

    static void Near(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(expected[i]) && float.IsFinite(actual[i]));
            Assert.True(Math.Abs((double)expected[i] - actual[i]) / Math.Max(1, Math.Abs(expected[i])) <= 1e-4,
                $"Projection mismatch at {i}: expected={expected[i]:R}, actual={actual[i]:R}");
        }
    }

    [Theory]
    [InlineData(1)] [InlineData(2)] [InlineData(3)] [InlineData(5)] [InlineData(8)] [InlineData(15)]
    [InlineData(25)] [InlineData(75)] [InlineData(128)] [InlineData(150)] [InlineData(600)]
    public void EveryVectorLaneMatchesAnIndependentComplexDft(int n)
    {
        int width = Vector<float>.Count;
        var real = new Vector<float>[n]; var imaginary = new Vector<float>[n];
        var outR = new Vector<float>[n]; var outI = new Vector<float>[n];
        var input = new Complex[n, width]; var rng = new Random(71 + n);
        for (int j = 0; j < n; j++)
        {
            var r = new float[width]; var i = new float[width];
            for (int lane = 0; lane < width; lane++)
            {
                r[lane] = (float)(rng.NextDouble() - .5); i[lane] = (float)(rng.NextDouble() - .5);
                input[j, lane] = new Complex(r[lane], i[lane]);
            }
            real[j] = new(r); imaginary[j] = new(i);
            outR[j] = new(float.NaN); outI[j] = new(float.NaN);
        }
        new FrameFourierTransform(n).Forward(real, imaginary, outR, outI);
        for (int k = 0; k < n; k++)
            for (int lane = 0; lane < width; lane++)
            {
                Complex expected = Complex.Zero;
                for (int j = 0; j < n; j++)
                {
                    double angle = -2 * Math.PI * ((long)j * k % n) / n;
                    expected += input[j, lane] * new Complex(Math.Cos(angle), Math.Sin(angle));
                }
                var actual = new Complex(outR[k][lane], outI[k][lane]);
                Assert.True(Complex.Abs(actual - expected) / Math.Max(1, Complex.Abs(expected)) <= 1e-4);
                Assert.Equal(input[k, lane], new Complex(real[k][lane], imaginary[k][lane]));
            }
    }

    [Theory]
    [InlineData(1)] [InlineData(7)] [InlineData(8)] [InlineData(9)] [InlineData(17)]
    public void ProjectionOverwritesDirtyStorageAndKeepsTailGuards(int frames)
    {
        const int n = 30, step = 7;
        var input = Input(n, frames, step); var weight = Weights(n);
        var expected = Reference(input, weight, step);
        var storage = Enumerable.Repeat(float.NaN, expected.Length + 8).ToArray();
        var rented = Enumerable.Range(0, 4).Select(_ => ArrayPool<Vector<float>>.Shared.Rent(n)).ToArray();
        foreach (var array in rented) Array.Fill(array, new Vector<float>(float.NaN));
        foreach (var array in rented) ArrayPool<Vector<float>>.Shared.Return(array);
        var scratch = new ScratchAccountant();
        var plan = PreparedFourierProjection.Hann(n);
        Assert.True(plan.TryWrite(input.Buffer.Span, weight.Buffer.Span, step, storage.AsSpan(4, expected.Length), scratch));
        Near(expected, storage.AsSpan(4, expected.Length).ToArray());
        Assert.All(storage.Take(4).Concat(storage.TakeLast(4)), value => Assert.True(float.IsNaN(value)));
        Assert.Equal(4L * n * Vector<float>.Count * sizeof(float), scratch.TotalScratchBytes);
        Assert.Equal(PreparedFourierProjection.RequiredBytes(n), plan.RetainedBytes);
    }

    [Fact]
    public void FullBitGuardRefusesTinyWeightChangesAndLeavesOutputUntouched()
    {
        var plan = PreparedFourierProjection.Hann(30); var weight = Weights(30); var input = Input(30, 9, 7);
        var output = Enumerable.Repeat(123f, 32 * 9).ToArray();
        int index = (int)weight.Length - 1;
        float original = weight.Buffer.Span[index];
        weight.Buffer.Span[index] = BitConverter.Int32BitsToSingle(BitConverter.SingleToInt32Bits(original) ^ 1);
        Assert.False(plan.Matches(weight.Buffer.Span));
        Assert.False(plan.TryWrite(input.Buffer.Span, weight.Buffer.Span, 7, output, null));
        Assert.All(output, value => Assert.Equal(123f, value));
        weight.Buffer.Span[index] = original;
        Assert.True(plan.Matches(weight.Buffer.Span));
        foreach (float invalid in new[] { float.NaN, float.PositiveInfinity, 1e30f, float.Epsilon })
        {
            float saved = input.Buffer.Span[3]; input.Buffer.Span[3] = invalid;
            Assert.False(plan.TryWrite(input.Buffer.Span, weight.Buffer.Span, 7, output, null));
            input.Buffer.Span[3] = saved;
            Assert.All(output, value => Assert.Equal(123f, value));
        }
    }

    static ComputationalGraph Graph(long cap = long.MaxValue, bool dynamic = false,
        Action<ComputationalGraph>? configure = null, ExecutionOptions? options = null)
    {
        const int n = 30, frames = 17, step = 7;
        var graph = new ComputationalGraph(cap) { Opset = new() { [""] = 17 } };
        graph.Metadata["Name"] = "private-Fourier-preparation";
        graph.Inputs["x"] = Input(n, frames, step);
        graph.Outputs["y"] = DenseTensor<float>.OfShape(1, 32, frames);
        if (!dynamic) graph.Initializers["w"] = Weights(n);
        else
        {
            var window = Enumerable.Range(0, n).Select(j => (float)(.5 - .5 * Math.Cos(2 * Math.PI * j / n))).ToArray();
            var phases = new float[16 * n];
            for (int k = 0; k < 16; k++) for (int j = 0; j < n; j++) phases[k * n + j] = ((k * j) % n) * (float)(2 * Math.PI / n);
            var data = new DenseTensor<float>(phases, [16, n]);
            var cos = ((Tensor<float>)CPUExecutionProvider.Cos(data, null).Outputs[0]).ToArray();
            var sin = ((Tensor<float>)CPUExecutionProvider.Sin(data, null).Outputs[0]).ToArray();
            var basis = cos.Concat(sin.Select(value => -value)).ToArray();
            graph.Initializers["basis"] = new DenseTensor<float>(basis, [32, n]);
            graph.Initializers["window"] = new DenseTensor<float>(window, [1, n]);
            graph.Initializers["axis"] = DenseTensor<long>.OfValues(new long[] { 1 });
            graph.Nodes.Add(new() { Name = "windowing", Op = OpType.Mul, Inputs = ["basis", "window"], Outputs = ["matrix"], Attributes = new(), OpsetVersion = 17 });
            graph.Nodes.Add(new() { Name = "weight", Op = OpType.Unsqueeze, Inputs = ["matrix", "axis"], Outputs = ["w"], Attributes = new(), OpsetVersion = 17 });
        }
        graph.Nodes.Add(new() { Name = "projection", Op = OpType.Conv, Inputs = ["x", "w"], Outputs = ["y"], OpsetVersion = 17,
            Attributes = new() { ["kernel_shape"] = new long[] { n }, ["strides"] = new long[] { step }, ["pads"] = new long[] { 0, 0 } } });
        graph.Options = options ?? ExecutionOptions.Default;
        configure?.Invoke(graph);
        graph.Prepare();
        return graph;
    }

    static Tensor<float> Run(ComputationalGraph graph)
    {
        graph.Reset();
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = Input(30, 17, 7) }, false), graph.LastErrorMessage);
        return (Tensor<float>)graph.Outputs["y"];
    }

    [SkippableFact]
    public void GraphPreparationUsesTheSharedCapAndInvalidatesWithoutExposingTemplates()
    {
        Skip.IfNot(Fma.IsSupported && Vector.IsHardwareAccelerated, "Prepared intrinsic projection requires hardware vectors.");
        long bytes = PreparedFourierProjection.RequiredBytes(30);
        var refused = Graph(bytes - 1); var graph = Graph(bytes);
        Assert.Null(refused.FourierProjections);
        Assert.Equal(0, refused.RetainedPackedWeightBytes);
        Assert.Single(graph.FourierProjections![30]);
        Assert.Equal(bytes, graph.RetainedPackedWeightBytes);
        Assert.Equal(new[] { "w" }, graph.Initializers.Keys);
        Near(Run(refused).ToArray(), Run(graph).ToArray());
        graph.InvalidatePreparation();
        Assert.Null(graph.FourierProjections);
        Assert.Equal(0, graph.RetainedPackedWeightBytes);
        graph.Prepare();
        Assert.Equal(bytes, graph.RetainedPackedWeightBytes);
    }

    [SkippableTheory]
    [InlineData(false)] [InlineData(true)]
    public void LiveSourcesFallbackAndHeldOutputsSurviveRepeatedRuns(bool dynamic)
    {
        Skip.IfNot(Fma.IsSupported && Vector.IsHardwareAccelerated, "Prepared intrinsic projection requires hardware vectors.");
        var graph = Graph(dynamic: dynamic); var control = Graph(0, dynamic);
        Assert.NotNull(graph.FourierProjections);
        var held = new List<(Tensor<float> Tensor, float[] Values)>();
        for (int repeat = 0; repeat < 4; repeat++)
        {
            foreach (var target in new[] { graph, control })
            {
                if (repeat == 1)
                {
                    var source = (DenseTensor<float>)target.Initializers[dynamic ? "window" : "w"];
                    source.Buffer.Span[3] += .125f;
                }
                if (repeat == 2)
                {
                    string name = dynamic ? "basis" : "w";
                    var old = (Tensor<float>)target.Initializers[name]; var values = old.ToArray(); values[5] += .25f;
                    target.Initializers[name] = new DenseTensor<float>(values, old.Dimensions.ToArray());
                }
            }
            var expected = Run(control).ToArray(); var actual = Run(graph);
            Near(expected, actual.ToArray());
            if (repeat == 1 || repeat == 2) Assert.Equal(expected, actual.ToArray());
            held.Add((actual, actual.ToArray()));
            foreach (var prior in held) Assert.Equal(prior.Values, prior.Tensor.ToArray());
        }
    }

    [SkippableFact]
    public async Task ExplicitContextsShareReadonlyPlansAndKeepTheirOwnOutputs()
    {
        Skip.IfNot(Fma.IsSupported && Vector.IsHardwareAccelerated, "Prepared intrinsic projection requires hardware vectors.");
        var graph = Graph(); var control = Graph(0); var expected = Run(control).ToArray();
        var contexts = Enumerable.Range(0, 4).Select(_ => graph.CreateExecution(null)).ToArray();
        await Task.WhenAll(contexts.Select(context => Task.Run(() =>
        {
            for (int repeat = 0; repeat < 3; repeat++) Near(expected, Run(context).ToArray());
        })));
        Assert.All(contexts, context => Assert.Same(graph.FourierProjections, context.FourierProjections));
        Assert.Equal(PreparedFourierProjection.RequiredBytes(30), graph.RetainedPackedWeightBytes);
    }

    [SkippableTheory]
    [InlineData(false)] [InlineData(true)]
    public void DeclaredFeedsOverridePreparedWeightsAndTheirGeneratingSources(bool dynamic)
    {
        Skip.IfNot(Fma.IsSupported && Vector.IsHardwareAccelerated, "Prepared intrinsic projection requires hardware vectors.");
        string name = dynamic ? "window" : "w";
        void Declare(ComputationalGraph graph) => graph.Inputs[name] = graph.Initializers[name];
        var graph = Graph(dynamic: dynamic, configure: Declare);
        var control = Graph(0, dynamic, Declare);
        Assert.NotNull(graph.FourierProjections);
        var replacement = (Tensor<float>)graph.Initializers[name];
        var values = replacement.ToArray(); values[4] += .25f;
        var feed = new DenseTensor<float>(values, replacement.Dimensions.ToArray());
        var inputs = new Dictionary<string, ITensor> { ["x"] = Input(30, 17, 7), [name] = feed };
        Assert.True(control.Execute(inputs, false), control.LastErrorMessage);
        Assert.True(graph.Execute(inputs, false), graph.LastErrorMessage);
        Assert.Equal(((Tensor<float>)control.Outputs["y"]).ToArray(), ((Tensor<float>)graph.Outputs["y"]).ToArray());
        Assert.Equal(values, feed.ToArray());
    }

    [SkippableTheory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)]
    public void ScalarPortableAndParallelContextsKeepTheirExistingArithmetic(int mode)
    {
        Skip.IfNot(Fma.IsSupported && Vector.IsHardwareAccelerated, "Prepared intrinsic projection requires hardware vectors.");
        var options = mode == 0 ? ExecutionOptions.Scalar : mode == 1 ? ExecutionOptions.Simd
            : ExecutionOptions.Default with { Tensor = TensorExecutionOptions.Parallel(2) };
        var graph = Graph();
        var context = graph.CreateExecution(options);
        var control = Graph(0, options: options);
        Assert.Equal(Run(control).ToArray(), Run(context).ToArray());
        Assert.Null(control.FourierProjections);
    }

    [SkippableTheory]
    [InlineData("padding")] [InlineData("dilation")]
    public void UnsupportedConvolutionGeometryUsesTheOriginalProvider(string feature)
    {
        Skip.IfNot(Fma.IsSupported && Vector.IsHardwareAccelerated, "Prepared intrinsic projection requires hardware vectors.");
        void Change(ComputationalGraph graph)
        {
            var node = graph.Nodes.Single();
            if (feature == "padding") node.Attributes["pads"] = new long[] { 1, 2 };
            if (feature == "dilation") node.Attributes["dilations"] = new long[] { 2 };
        }
        var graph = Graph(configure: Change); var control = Graph(0, configure: Change);
        Assert.Null(graph.FourierProjections);
        Assert.Equal(Run(control).ToArray(), Run(graph).ToArray());
    }

    [SkippableFact]
    public void ProviderBiasKeepsTheExistingReductionEvenWithAnAvailableRecipe()
    {
        Skip.IfNot(Fma.IsSupported && Vector.IsHardwareAccelerated, "Prepared intrinsic projection requires hardware vectors.");
        var input = Input(30, 17, 7); var weight = Weights(30);
        var bias = DenseTensor<float>.OfValues(Enumerable.Repeat(.25f, 32).ToArray());
        var graph = Graph();
        float[] Execute(ExecutionOptions options)
        {
            var result = CPUExecutionProvider.Conv(input, weight, bias, null, [1], 1, [30], [0, 0], [7], options, null);
            Assert.Equal(OpStatus.Success, result.Status);
            return ((Tensor<float>)Assert.Single(result.Outputs)).ToArray();
        }
        var enabled = ExecutionOptions.Default with { Tensor = ExecutionOptions.Default.Tensor with { FourierProjections = graph.FourierProjections } };
        Assert.Equal(Execute(ExecutionOptions.Default), Execute(enabled));
    }

    [SkippableTheory]
    [InlineData(false)] [InlineData(true)]
    public void ReversedAndOffsetWeightStorageUsesItsLogicalValues(bool reverse)
    {
        Skip.IfNot(Fma.IsSupported && Vector.IsHardwareAccelerated, "Prepared intrinsic projection requires hardware vectors.");
        void Change(ComputationalGraph graph)
        {
            var weight = Weights(30); var values = weight.ToArray();
            if (reverse) graph.Initializers["w"] = new DenseTensor<float>(values.AsMemory(), [32, 1, 30], reverseStride: true);
            else
            {
                var storage = Enumerable.Repeat(123f, values.Length + 2).ToArray();
                values.CopyTo(storage, 1);
                graph.Initializers["w"] = new DenseTensor<float>(storage.AsMemory(1, values.Length), [32, 1, 30]);
            }
        }
        var graph = Graph(configure: Change); var control = Graph(0, configure: Change);
        if (reverse) Assert.Null(graph.FourierProjections);
        Near(Run(control).ToArray(), Run(graph).ToArray());
    }

    [SkippableTheory]
    [InlineData(0)] [InlineData(1)]
    public void FourierTemplatesShareTheAggregateBudgetWithMatrixPanels(int extraBytes)
    {
        Skip.IfNot(Fma.IsSupported && Vector.IsHardwareAccelerated, "Prepared intrinsic projection requires hardware vectors.");
        long bytes = PreparedFourierProjection.RequiredBytes(30);
        void AddMatrix(ComputationalGraph graph)
        {
            graph.Initializers["matrix"] = DenseTensor<float>.OfValues(new float[,] { { 1, 2 }, { 3, 4 } });
            graph.Inputs["a"] = DenseTensor<float>.OfValues(new float[,] { { 1, 2 } });
            graph.Outputs["z"] = DenseTensor<float>.OfShape(1, 2);
            graph.Nodes.Add(new() { Name = "matrix-product", Op = OpType.MatMul, Inputs = ["a", "matrix"], Outputs = ["z"], Attributes = new() });
        }
        var graph = Graph(bytes + 15 + extraBytes, configure: AddMatrix);
        Assert.Single(graph.PackedWeights);
        Assert.Equal(extraBytes == 1, graph.FourierProjections is not null);
        Assert.Equal(16 + (extraBytes == 1 ? bytes : 0), graph.RetainedPackedWeightBytes);
        graph.RefreshLifetimeAnalysis();
        Assert.Equal(16 + (extraBytes == 1 ? bytes : 0), graph.RetainedPackedWeightBytes);
    }

    [SkippableFact]
    public void MultipleConsumersShareOnePrivateRecipe()
    {
        Skip.IfNot(Fma.IsSupported && Vector.IsHardwareAccelerated, "Prepared intrinsic projection requires hardware vectors.");
        var graph = Graph(configure: graph =>
        {
            graph.Outputs["z"] = DenseTensor<float>.OfShape(1, 32, 17);
            graph.Nodes.Add(new() { Name = "second-projection", Op = OpType.Conv, Inputs = ["x", "w"], Outputs = ["z"],
                Attributes = new() { ["kernel_shape"] = new long[] { 30 }, ["strides"] = new long[] { 7 }, ["pads"] = new long[] { 0, 0 } }, OpsetVersion = 17 });
        });
        Assert.Single(graph.FourierProjections![30]);
        Assert.Equal(PreparedFourierProjection.RequiredBytes(30), graph.RetainedPackedWeightBytes);
        var output = Run(graph);
        Assert.Equal(output.ToArray(), ((Tensor<float>)graph.Outputs["z"]).ToArray());
    }
}
