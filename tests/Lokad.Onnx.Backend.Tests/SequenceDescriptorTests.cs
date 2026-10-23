using Google.Protobuf;
using Onnx;

namespace Lokad.Onnx.Backend.Tests;

public class SequenceDescriptorTests
{
    static TypeProto SequenceType(TypeProto? element) => new()
    {
        SequenceType = new TypeProto.Types.Sequence { ElemType = element }
    };

    static TypeProto TensorType(int dtype = 1, bool unknownRank = false) => new()
    {
        TensorType = new TypeProto.Types.Tensor
        {
            ElemType = dtype,
            Shape = unknownRank ? null : new TensorShapeProto { Dim = { new TensorShapeProto.Types.Dimension() } }
        }
    };

    static byte[] IdentityModel(TypeProto input, TypeProto output)
    {
        var model = new ModelProto
        {
            IrVersion = 9, OpsetImport = { new OperatorSetIdProto { Version = 18 } },
            Graph = new GraphProto
            {
                Name = "sequence-types",
                Input = { new ValueInfoProto { Name = "x", Type = input } },
                Output = { new ValueInfoProto { Name = "y", Type = output } },
                Node = { new NodeProto { Name = "copy", OpType = "Identity", Input = { "x" }, Output = { "y" } } }
            }
        };
        return model.ToByteArray();
    }

    [Theory]
    [InlineData(false, false)] [InlineData(false, true)]
    [InlineData(true, false)] [InlineData(true, true)]
    public void ImportedFloatSequenceRejectsWrongAndMixedItemsAndAllowsRetry(bool positional, bool context)
    {
        var type = SequenceType(TensorType());
        var plan = OnnxImport.Load(IdentityModel(type, type.Clone()))!;
        Assert.NotNull(plan);
        Assert.Equal(TensorElementType.Float, plan.InputDescs[0].SequenceElementType);
        Assert.Equal(TensorElementType.Float, plan.OutputDescs[0].SequenceElementType);
        ComputationalGraph graph = context ? plan.CreateExecution(ExecutionOptions.Memory) : plan;
        bool Run(TensorSequence value) => graph.Execute(positional ? new ITensor[] { value }
            : new Dictionary<string, ITensor> { ["x"] = value }, false);
        var item = DenseTensor<float>.OfValues([1, 2]);
        var good = new TensorSequence(new ITensor[] { item });
        Assert.True(Run(good), graph.LastErrorMessage);
        var held = Assert.IsType<TensorSequence>(graph.Outputs["y"]);
        foreach (var bad in new[]
        {
            new TensorSequence(new ITensor[] { DenseTensor<int>.Scalar(3) }),
            new TensorSequence(new ITensor[] { item, DenseTensor<int>.Scalar(3) }),
            new TensorSequence(new ITensor[] { new TensorSequence() })
        })
        {
            Assert.False(Run(bad));
            Assert.Contains("sequence element", graph.LastErrorMessage);
            Assert.Empty(graph.Outputs);
            Assert.True(Run(good), graph.LastErrorMessage);
        }
        graph.Reset();
        Assert.True(Run(new TensorSequence()), graph.LastErrorMessage);
        Assert.Empty(Assert.IsType<TensorSequence>(graph.Outputs["y"]).Items);
        Assert.Equal(new float[] { 1, 2 }, ((Tensor<float>)Assert.Single(held.Items)).ToArray());
        Assert.Equal(new float[] { 1, 2 }, item.ToArray());
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void FloatElementsMayHaveVaryingShapesAndUnknownRank(bool unknownRank)
    {
        var type = SequenceType(TensorType(unknownRank: unknownRank));
        var graph = OnnxImport.Load(IdentityModel(type, type.Clone()))!;
        ITensor[] items = unknownRank
            ? [DenseTensor<float>.Scalar(1), DenseTensor<float>.OfValues(new float[,] { { 2, 3 } })]
            : [DenseTensor<float>.OfValues([1]), DenseTensor<float>.OfValues([2, 3])];
        Assert.True(graph.Execute(new ITensor[] { new TensorSequence(items) }, false), graph.LastErrorMessage);
        var clone = (TensorSequence)graph.Outputs["y"].Clone();
        for (int i = 0; i < items.Length; i++)
        {
            Assert.NotSame(items[i], clone.Items[i]);
            Assert.Equal(items[i].Dims, clone.Items[i].Dims);
            Assert.Equal(((Tensor<float>)items[i]).ToArray(), ((Tensor<float>)clone.Items[i]).ToArray());
        }
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void ProducedSequenceMustMatchOutputElementType(bool context)
    {
        var plan = OnnxImport.Load(IdentityModel(SequenceType(TensorType(6)), SequenceType(TensorType())))!;
        ComputationalGraph graph = context ? plan.CreateExecution(ExecutionOptions.Scalar) : plan;
        Assert.False(graph.Execute(new ITensor[] { new TensorSequence(new ITensor[] { DenseTensor<int>.Scalar(3) }) }, false));
        Assert.Contains("Graph output y", graph.LastErrorMessage);
        Assert.Contains("sequence element", graph.LastErrorMessage);
        Assert.Empty(graph.Outputs);
        Assert.True(graph.Execute(new ITensor[] { new TensorSequence() }, false), graph.LastErrorMessage);
    }

    [Theory]
    [InlineData("missing")] [InlineData("map")] [InlineData("optional")] [InlineData("nested")]
    [InlineData("undefined")] [InlineData("unsupported")]
    public void UnsupportedSequenceElementsFailExplicitlyAtImport(string kind)
    {
        TypeProto? element = kind switch
        {
            "missing" => null,
            "map" => new TypeProto { MapType = new TypeProto.Types.Map() },
            "optional" => new TypeProto { OptionalType = new TypeProto.Types.Optional() },
            "nested" => SequenceType(TensorType()),
            "undefined" => TensorType(0),
            _ => TensorType(999)
        };
        var value = new ValueInfoProto { Name = "x", Type = SequenceType(element) };
        var error = Assert.Throws<NotSupportedException>(() => value.ToValueDto());
        Assert.Contains("sequence", error.Message);
        Assert.Contains("x", error.Message);
    }

    [Fact]
    public void LegacyUntypedDescriptionsKeepGeneralSequenceBehavior()
    {
        var model = new OnnxModel
        {
            Inputs = { new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Sequence } },
            Outputs = { new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Sequence } }
        };
        var graph = Model.Load(model);
        var sequence = new TensorSequence(new ITensor[] { DenseTensor<float>.Scalar(1), DenseTensor<int>.Scalar(2) });
        Assert.True(graph.Execute(new ITensor[] { sequence }, false), graph.LastErrorMessage);
        Assert.Null(graph.InputDescs[0].SequenceElementType);
        var output = Assert.IsType<TensorSequence>(graph.Outputs["x"]);
        Assert.Equal(new[] { TensorElementType.Float, TensorElementType.Int32 }, output.Items.Select(item => item.ElementType));
        Assert.NotSame(sequence, output);
    }
}
