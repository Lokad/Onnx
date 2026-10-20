namespace Lokad.Onnx.Backend.Tests;

public class GraphFusionRmsTests
{
    static OnnxValueInfo IO(string name, int[] shape) => new() { Name = name, Dims = shape, ElementType = TensorElementType.Float };
    static OnnxTensor Float(string name, int[] shape, float[] values) => new() { Name = name, Dims = shape, Data = values, ElementType = TensorElementType.Float };
    static OnnxNode Node(string op, string[] inputs, string output) => new() {
        OpType = op, Domain = "", Inputs = inputs, Outputs = new[] { output }, Attributes = new Dictionary<string, object>() };

    static OnnxModel ModelDefinition(int rows, int width, string exposed) => ModelDefinition(rows, width, exposed, new[] { rows, width });

    static OnnxModel ModelDefinition(int rows, int width, string exposed, int[] outputShape)
    {
        var model = new OnnxModel { Name = "rms-test" };
        model.Opset[""] = 18;
        model.Inputs.Add(IO("x", new[] { rows, width }));
        model.Outputs.Add(IO("y", outputShape));
        model.Initializers.Add(Float("epsilon", Array.Empty<int>(), new[] { 1e-6f }));
        model.Initializers.Add(Float("exponent", Array.Empty<int>(), new[] { -.5f }));
        model.Initializers.Add(Float("scale", new[] { width }, Enumerable.Range(0, width).Select(i => (i % 7 - 3) * .125f).ToArray()));
        model.Initializers.Add(new OnnxTensor { Name = "axes", Dims = new[] { 1 }, Data = new[] { -1L }, ElementType = TensorElementType.Int64 });
        model.Nodes.Add(Node("Mul", new[] { "x", "x" }, "square"));
        var mean = Node("ReduceMean", new[] { "square", "axes" }, "mean");
        mean.Attributes["keepdims"] = 1;
        model.Nodes.Add(mean);
        model.Nodes.Add(Node("Add", new[] { "mean", "epsilon" }, "shifted"));
        model.Nodes.Add(Node("Pow", new[] { "shifted", "exponent" }, "inverse"));
        model.Nodes.Add(Node("Mul", new[] { "x", "inverse" }, "normalized"));
        model.Nodes.Add(Node("Mul", new[] { "normalized", "scale" }, "y"));
        if (exposed.Length != 0) model.Outputs.Add(IO(exposed,
            new[] { rows, exposed is "square" or "normalized" ? width : 1 }));
        return model;
    }

    static DenseTensor<float> Input(int rows, int width)
    {
        var result = new DenseTensor<float>(new[] { rows, width });
        for (int i = 0; i < result.Length; i++) result.SetValue(i, (i % 23 - 11) * .0625f);
        return result;
    }

    static int[] Run(ComputationalGraph graph, ITensor input, ExecutionOptions options)
    {
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true, ExecutionProvider.CPU, options), graph.LastErrorMessage);
        return ((Tensor<float>)graph.Outputs["y"]).ToArray().Select(BitConverter.SingleToInt32Bits).ToArray();
    }

    [Fact]
    public void CanonicalPatternRetainsEveryRuntimeControlAndExactOutputs()
    {
        var graph = Model.Load(ModelDefinition(5, 33, ""))!;
        var node = Assert.Single(graph.Nodes);
        Assert.Equal(OpType.FusedRmsNormalization, node.Op);
        Assert.True(node.IsFused);
        Assert.Equal(new[] { "x", "axes", "epsilon", "exponent", "scale" }, node.Inputs);
        Assert.True(CPUExecutionProvider.SupportsNode(node));
        var reference = Model.Load(ModelDefinition(5, 33, "square"))!;
        foreach (bool simd in new[] { false, true })
        {
            var options = new ExecutionOptions(OptimizationMode.Memory, new TensorExecutionOptions(simd, false, 1));
            var input = Input(5, 33);
            var before = input.ToArray();
            Assert.Equal(Run(reference, input, options), Run(graph, input, options));
            Assert.Equal(before, input.ToArray());
        }
    }

    [Fact]
    public void MissingGraphImportPreservesExplicitNodeOpset()
    {
        var graph = Model.Load(ModelDefinition(5, 33, ""), runOptimizer: false)!;
        var reference = Model.Load(ModelDefinition(5, 33, "square"))!;
        Assert.All(graph.Nodes, node => Assert.Equal(18, node.OpsetVersion));
        graph.Opset.Clear();
        Optimization.GraphOptimizer.Run(graph);
        var fused = Assert.Single(graph.Nodes);
        Assert.Equal(OpType.FusedRmsNormalization, fused.Op);
        Assert.Equal(18, fused.OpsetVersion);
        Assert.True(CPUExecutionProvider.SupportsNode(fused));
        Assert.Equal(Run(reference, Input(5, 33), ExecutionOptions.Memory),
            Run(graph, Input(5, 33), ExecutionOptions.Memory));
    }

    [Fact]
    public void MutatedInitializersFallBackWithoutBakingValuesOrShapes()
    {
        var graph = Model.Load(ModelDefinition(5, 33, ""))!;
        var reference = Model.Load(ModelDefinition(5, 33, "square"))!;
        var input = Input(5, 33);
        var options = ExecutionOptions.Memory;
        Assert.Equal(Run(reference, input, options), Run(graph, input, options));
        foreach (string control in new[] { "axes", "epsilon", "exponent", "scale" })
        {
            ITensor changed = control switch {
                "axes" => DenseTensor<long>.OfValues(new[] { 0L }),
                "epsilon" => new DenseTensor<float>(new Memory<float>(new[] { -100f }), Array.Empty<int>()),
                "exponent" => new DenseTensor<float>(new Memory<float>(new[] { 2f }), Array.Empty<int>()),
                _ => DenseTensor<float>.OfValues(Enumerable.Repeat(-.25f, 33).ToArray()) };
            graph.Initializers[control] = changed;
            reference.Initializers[control] = changed;
            Assert.Equal(Run(reference, input, options), Run(graph, input, options));
        }
        // A scalar control may acquire a leading dimension. Its output shape is not X's shape.
        // Declare that expanded output rank so the ordinary descriptor guard also remains satisfied.
        graph = Model.Load(ModelDefinition(5, 33, "", new[] { 1, 5, 33 }))!;
        reference = Model.Load(ModelDefinition(5, 33, "square", new[] { 1, 5, 33 }))!;
        var epsilon = new DenseTensor<float>(new Memory<float>(Enumerable.Repeat(1e-6f, 5).ToArray()), new[] { 1, 5, 1 });
        graph.Initializers["epsilon"] = epsilon;
        reference.Initializers["epsilon"] = epsilon;
        Assert.Equal(Run(reference, input, options), Run(graph, input, options));
        Assert.Equal(new[] { 1, 5, 33 }, graph.Outputs["y"].Dims);
        Assert.False(Optimization.GraphFacts.Build(graph).KnownDims.ContainsKey("y"));
    }

    [Theory]
    [InlineData("square")]
    [InlineData("mean")]
    [InlineData("shifted")]
    [InlineData("inverse")]
    [InlineData("normalized")]
    public void ExposedOrSharedIntermediateRefusesFusion(string name)
    {
        var exposed = Model.Load(ModelDefinition(5, 33, name))!;
        Assert.DoesNotContain(exposed.Nodes, n => n.Op == OpType.FusedRmsNormalization);
        var definition = ModelDefinition(5, 33, "");
        definition.Nodes.Add(Node("Add", new[] { name, "epsilon" }, "side"));
        definition.Outputs.Add(IO("side", Array.Empty<int>()));
        var shared = Model.Load(definition)!;
        Assert.DoesNotContain(shared.Nodes, n => n.Op == OpType.FusedRmsNormalization);
    }

    [Theory]
    [InlineData("square")]
    [InlineData("mean")]
    [InlineData("shifted")]
    [InlineData("inverse")]
    [InlineData("normalized")]
    public void NestedCaptureRefusesFusionAndRetainsExecution(string name)
    {
        var definition = ModelDefinition(5, 33, "");
        int width = name is "square" or "normalized" ? 33 : 1;
        OnnxSubgraph Branch() => new() { Outputs = { IO(name, new[] { 5, width }) } };
        definition.Inputs.Add(new OnnxValueInfo { Name = "condition", ElementType = TensorElementType.Bool, Dims = Array.Empty<int>() });
        definition.Outputs.Add(IO("captured", new[] { 5, width }));
        definition.Nodes.Add(new OnnxNode { OpType = "If", Inputs = new[] { "condition" }, Outputs = new[] { "captured" },
            Attributes = new Dictionary<string, object> { ["then_branch"] = Branch(), ["else_branch"] = Branch() } });
        var graph = Model.Load(definition)!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.FusedRmsNormalization);
        Assert.Contains(name, Optimization.GraphFacts.Build(graph).CapturedValues);
        Assert.True(graph.Execute(new Dictionary<string, ITensor> {
            ["x"] = Input(5, 33), ["condition"] = new DenseTensor<bool>(new[] { true }, Array.Empty<int>()) }, true), graph.LastErrorMessage);
        Assert.Equal(new[] { 5, width }, graph.Outputs["captured"].Dims);
    }

    [Fact]
    public void RuntimeControlInputsRemainLiveAcrossCalls()
    {
        OnnxModel Definition(string exposed)
        {
            var definition = ModelDefinition(5, 33, exposed);
            foreach (var initializer in definition.Initializers)
                definition.Inputs.Add(new OnnxValueInfo { Name = initializer.Name, Dims = initializer.Dims, ElementType = initializer.ElementType });
            definition.Initializers.Clear();
            return definition;
        }
        var graph = Model.Load(Definition(""))!;
        var reference = Model.Load(Definition("square"))!;
        Assert.Contains(graph.Nodes, n => n.Op == OpType.FusedRmsNormalization);
        var feed = new Dictionary<string, ITensor> {
            ["x"] = Input(5, 33), ["axes"] = DenseTensor<long>.OfValues(new[] { 0L }),
            ["epsilon"] = new DenseTensor<float>(new[] { .125f }, Array.Empty<int>()),
            ["exponent"] = new DenseTensor<float>(new[] { 2f }, Array.Empty<int>()),
            ["scale"] = DenseTensor<float>.OfValues(Enumerable.Repeat(.75f, 33).ToArray()) };
        for (int run = 0; run < 2; run++)
        {
            Assert.True(graph.Execute(feed, true), graph.LastErrorMessage);
            Assert.True(reference.Execute(feed, true), reference.LastErrorMessage);
            Assert.Equal(((Tensor<float>)reference.Outputs["y"]).ToArray().Select(BitConverter.SingleToInt32Bits),
                ((Tensor<float>)graph.Outputs["y"]).ToArray().Select(BitConverter.SingleToInt32Bits));
            feed["axes"] = DenseTensor<long>.OfValues(new[] { -1L });
            feed["exponent"] = new DenseTensor<float>(new[] { -.5f }, Array.Empty<int>());
            ((Tensor<float>)feed["scale"])[0] = -.5f;
        }
    }

    [Fact]
    public void DoubleAndStridedFallbacksRetainPrimitiveSemantics()
    {
        var input = new DenseTensor<double>(new[] { 5, 33 });
        var expected = new double[165];
        for (int i = 0; i < input.Length; i++) input.SetValue(i, (i % 17 - 8) * .0625);
        var axes = DenseTensor<long>.OfValues(new[] { -1L });
        var epsilon = new DenseTensor<double>(new[] { 1e-6 }, Array.Empty<int>());
        var exponent = new DenseTensor<double>(new[] { -.5 }, Array.Empty<int>());
        var scale = DenseTensor<double>.OfValues(Enumerable.Repeat(.75, 33).ToArray());
        for (int row = 0; row < 5; row++)
        {
            double sum = 0;
            for (int column = 0; column < 33; column++) sum += input[row, column] * input[row, column];
            double inverse = Math.Pow(sum / 33 + 1e-6, -.5);
            for (int column = 0; column < 33; column++) expected[row * 33 + column] = (input[row, column] * inverse) * .75;
        }
        var result = CPUExecutionProvider.FusedRmsNormalization(input, axes, epsilon, exponent, scale, 1, 0, ExecutionOptions.Memory, new TensorBufferPool());
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(expected.Select(BitConverter.DoubleToInt64Bits), ((Tensor<double>)Assert.Single(result.Outputs!)).ToArray().Select(BitConverter.DoubleToInt64Bits));

        var storage = Input(10, 66);
        var view = storage.Slice(new SliceIndex(0, 10, 2), new SliceIndex(0, 66, 2));
        var graph = Model.Load(ModelDefinition(5, 33, ""))!;
        var reference = Model.Load(ModelDefinition(5, 33, "square"))!;
        Assert.Equal(Run(reference, view, ExecutionOptions.Memory), Run(graph, view, ExecutionOptions.Memory));
    }

    [Fact]
    public void ExceptionalAndZeroEpsilonFallbackRetainsPrimitiveBits()
    {
        foreach (float value in new[] { 0f, -0f, float.PositiveInfinity, float.NegativeInfinity,
                     BitConverter.Int32BitsToSingle(0x7fa12345), BitConverter.Int32BitsToSingle(unchecked((int)0xffc54321)) })
        foreach (bool simd in new[] { false, true })
        {
            var graph = Model.Load(ModelDefinition(5, 33, ""))!;
            var reference = Model.Load(ModelDefinition(5, 33, "square"))!;
            var input = Input(5, 33);
            input[0, 0] = value;
            input[4, 32] = value;
            var epsilon = new DenseTensor<float>(new[] { 0f }, Array.Empty<int>());
            graph.Initializers["epsilon"] = epsilon;
            reference.Initializers["epsilon"] = epsilon;
            var options = new ExecutionOptions(OptimizationMode.Memory, new TensorExecutionOptions(simd, false, 1));
            Assert.Equal(Run(reference, input, options), Run(graph, input, options));
        }
    }

    [Fact]
    public void DomainAndInternalAdmissionGuardsRemainStrict()
    {
        var definition = ModelDefinition(5, 33, "");
        definition.Opset["vendor"] = 1;
        definition.Nodes[3].Domain = "vendor";
        var graph = Model.Load(definition)!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.FusedRmsNormalization);
        var node = new Lokad.Onnx.Node { Op = OpType.FusedRmsNormalization, OpTypeName = "FusedRmsNormalization",
            Domain = "", OpsetVersion = 18, Inputs = new[] { "x", "axes", "epsilon", "exponent", "scale" }, Outputs = new[] { "y" } };
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        node.IsFused = true;
        Assert.True(CPUExecutionProvider.SupportsNode(node));
        node.Domain = "vendor";
        Assert.False(CPUExecutionProvider.SupportsNode(node));
    }

    [Fact]
    public void PooledFastAndFallbackOutputsRemainOwnedAndPrivateTempsReturn()
    {
        var pool = new TensorBufferPool();
        var input = Input(5, 33);
        var axes = DenseTensor<long>.OfValues(new[] { -1L });
        var epsilon = new DenseTensor<float>(new Memory<float>(new[] { 1e-6f }), Array.Empty<int>());
        var exponent = new DenseTensor<float>(new Memory<float>(new[] { -.5f }), Array.Empty<int>());
        var scale = DenseTensor<float>.OfValues(Enumerable.Repeat(.75f, 33).ToArray());
        ITensor Evaluate(ITensor value, ITensor axis) {
            var result = CPUExecutionProvider.FusedRmsNormalization(value, axis, epsilon, exponent, scale, 1, 0, ExecutionOptions.Memory, pool);
            Assert.Equal(OpStatus.Success, result.Status);
            return Assert.Single(result.Outputs!);
        }
        var first = (Tensor<float>)Evaluate(input, axes);
        var before = first.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray();
        input[0, 0] = BitConverter.Int32BitsToSingle(0x7fa12345);
        int returned = pool.Returned;
        var second = (Tensor<float>)Evaluate(input, axes);
        Assert.True(pool.Returned > returned);
        Assert.Equal(before, first.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray());
        Assert.True(float.IsNaN(second[0, 0]));
        returned = pool.Returned;
        Assert.ThrowsAny<ArgumentException>(() => Evaluate(input, DenseTensor<long>.OfValues(new[] { 1000L })));
        Assert.True(pool.Returned > returned);
        input[0, 0] = 1;
        var recovered = (Tensor<float>)Evaluate(input, axes);
        Assert.All(recovered.ToArray(), v => Assert.True(float.IsFinite(v)));
        Assert.Equal(before, first.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray());
    }
}
