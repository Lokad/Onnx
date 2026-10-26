using Google.Protobuf;
using Onnx;

namespace Lokad.Onnx.Backend.Tests;

public class ShapeDescriptorTests
{
    static ValueInfoProto Value(string name, bool emptySymbol, bool twoUnknownAxes = false)
    {
        var shape = new TensorShapeProto();
        shape.Dim.Add(emptySymbol ? new TensorShapeProto.Types.Dimension { DimParam = "" }
            : new TensorShapeProto.Types.Dimension());
        shape.Dim.Add(twoUnknownAxes ? new TensorShapeProto.Types.Dimension { DimParam = "" }
            : new TensorShapeProto.Types.Dimension { DimValue = 3 });
        return new ValueInfoProto { Name = name, Type = new TypeProto
            { TensorType = new TypeProto.Types.Tensor { ElemType = 1, Shape = shape } } };
    }

    static ComputationalGraph Identity(ValueInfoProto input, ValueInfoProto output, bool context)
    {
        var model = new ModelProto { IrVersion = 10,
            OpsetImport = { new OperatorSetIdProto { Version = 18 } },
            Graph = new GraphProto { Name = "unknown-dimensions", Input = { input }, Output = { output },
                Node = { new NodeProto { OpType = "Identity", Input = { "x" }, Output = { "y" } } } } };
        var graph = Model.Load(OnnxImport.Parse(model.ToByteArray()));
        return context ? graph.CreateExecution(ExecutionOptions.Scalar) : graph;
    }

    static void CheckRun(ComputationalGraph graph, int rows, int columns)
    {
        var data = Enumerable.Range(0, rows * columns).Select(value => (float)value).ToArray();
        var input = new DenseTensor<float>(data, new[] { rows, columns });
        Assert.True(graph.Execute(new ITensor[] { input }, false, ExecutionProvider.CPU, ExecutionOptions.Scalar), graph.LastErrorMessage);
        var output = Assert.IsAssignableFrom<Tensor<float>>(graph.Outputs["y"]);
        Assert.Equal(new[] { rows, columns }, output.Dimensions.ToArray());
        Assert.Equal(data, output.ToArray());
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void EmptySymbolOnInputOrOutput_AcceptsChangingBatchSizes(bool outputSymbol, bool context)
    {
        var graph = Identity(Value("x", !outputSymbol), Value("y", outputSymbol), context);
        foreach (int rows in new[] { 2, 0, 5 }) CheckRun(graph, rows, 3);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EmptySymbols_DoNotCoupleUnrelatedAxes(bool context)
    {
        var graph = Identity(Value("x", true, true), Value("y", true, true), context);
        CheckRun(graph, 2, 3);
        CheckRun(graph, 4, 1);
        CheckRun(graph, 0, 2);
    }

    public static TheoryData<int[], string?[]?, string> Labels => new()
    {
        { new[] { 0, 3 }, new string?[] { "batch", null }, "batchx3" },
        { new[] { -1, 3 }, null, "?x3" },
        { new[] { 2, -1, 0 }, new string?[] { null, null, null }, "2x?x0" },
        { new[] { 0, 3 }, new string?[] { null, null }, "0x3" },
        { new[] { 2, 3 }, null, "2x3" },
        { Array.Empty<int>(), null, "" },
        { new[] { 0, -1, 3 }, new string?[] { "batch" }, "batchx?x3" },
        { new[] { 99, 3 }, new string?[] { "n", null }, "nx3" },
    };

    [Theory]
    [MemberData(nameof(Labels))]
    public void Describe_DistinguishesSymbolsUnknownsAndFixedExtents(int[] dims, string?[]? symbols, string expectedShape)
    {
        var description = new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float,
            Dims = dims, DimParams = symbols };
        Assert.Equal("x:float:" + expectedShape, description.Describe());
    }
}
