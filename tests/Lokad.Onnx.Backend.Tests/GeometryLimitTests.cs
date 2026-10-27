using Google.Protobuf;
using Onnx;

namespace Lokad.Onnx.Backend.Tests;

public class GeometryLimitTests
{
    const long Bound = 1L << 48;
    static DenseTensor<long> Scalar(long value) => new(new[] { value }, Array.Empty<int>());

    [Fact]
    public void OversizedProviderRangeFailsAtTheBoundary()
    {
        Assert.ThrowsAny<ArgumentException>(() => CPUExecutionProvider.Range(Scalar(Bound + 1), Scalar(Bound + 1), Scalar(1), null));
        Assert.Equal(new[] { Bound - 1 }, ((Tensor<long>)CPUExecutionProvider.Range(Scalar(Bound - 1), Scalar(Bound), Scalar(1), null).Outputs[0]).ToArray());
        var graph = new ComputationalGraph();
        graph.Inputs["s"] = Scalar(Bound + 1);
        graph.Inputs["l"] = Scalar(Bound + 1);
        graph.Inputs["d"] = Scalar(1);
        var node = new Node { Op = OpType.Range, OpTypeName = "Range", OpsetVersion = 13, Inputs = new[] { "s", "l", "d" }, Outputs = new[] { "y" } };
        Assert.Equal(OpStatus.Failure, node.Execute(graph, ExecutionProvider.CPU, null).Status);
    }

    [Fact]
    public void SliceKeepsSemanticEndpointMarkersButRejectsOrdinaryOversizedGeometry()
    {
        var input = DenseTensor<int>.OfValues(new[] { 1, 2, 3 });
        var axis = DenseTensor<long>.OfValues(new[] { 0L });
        ITensor Values(long value) => DenseTensor<long>.OfValues(new[] { value });
        var result = CPUExecutionProvider.Slice(input, Values(long.MinValue), Values(long.MaxValue), axis, Values(1), null);
        Assert.Equal(new[] { 1, 2, 3 }, ((Tensor<int>)result.Outputs[0]).ToArray());
        var reverse = CPUExecutionProvider.Slice(input, Values(long.MaxValue), Values(long.MinValue), axis, Values(-1), null);
        Assert.Equal(new[] { 3, 2, 1 }, ((Tensor<int>)reverse.Outputs[0]).ToArray());
        Assert.ThrowsAny<ArgumentException>(() => CPUExecutionProvider.Slice(input, Values(Bound + 1), Values(Bound + 1), axis, Values(1), null));
        Assert.ThrowsAny<ArgumentException>(() => CPUExecutionProvider.Slice(input, Values(0), Values(-long.MaxValue + 1), axis, Values(-1), null));
        Assert.ThrowsAny<ArgumentException>(() => CPUExecutionProvider.Slice(input, Values(0), Values(3), axis, Values(long.MaxValue), null));
        Assert.ThrowsAny<ArgumentException>(() => CPUExecutionProvider.Slice(input, Values(0), Values(3), axis, Values(-long.MaxValue), null));
    }

    [Theory]
    [InlineData(long.MinValue)]
    [InlineData(-long.MaxValue)]
    public void ExporterReverseSliceEndMarkersWorkAtProviderAndGraphBoundaries(long end)
    {
        var input = DenseTensor<float>.OfValues(new[] { 1f, 2f, 3f });
        var starts = DenseTensor<long>.OfValues(new[] { -1L });
        var ends = DenseTensor<long>.OfValues(new[] { end });
        var axes = DenseTensor<long>.OfValues(new[] { 0L });
        var steps = DenseTensor<long>.OfValues(new[] { -1L });
        var result = CPUExecutionProvider.Slice(input, starts, ends, axes, steps, null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new[] { 3f, 2f, 1f }, ((Tensor<float>)result.Outputs[0]).ToArray());

        var model = new OnnxModel
        {
            Opset = { [""] = 13 },
            Inputs = { new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = [3] } },
            Outputs = { new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = [3] } },
            Initializers = {
                new OnnxTensor { Name = "start", ElementType = TensorElementType.Int64, Dims = [1], Data = new long[] { -1 } },
                new OnnxTensor { Name = "end", ElementType = TensorElementType.Int64, Dims = [1], Data = new[] { end } },
                new OnnxTensor { Name = "axis", ElementType = TensorElementType.Int64, Dims = [1], Data = new long[] { 0 } },
                new OnnxTensor { Name = "step", ElementType = TensorElementType.Int64, Dims = [1], Data = new long[] { -1 } }
            },
            Nodes = { new OnnxNode { OpType = "Slice", Name = "reverse", Inputs = ["x", "start", "end", "axis", "step"], Outputs = ["y"] } }
        };
        var graph = Model.Load(model);
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true), graph.LastErrorMessage);
        Assert.Equal(new[] { 3f, 2f, 1f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
        Assert.Equal(new[] { 1f, 2f, 3f }, input.ToArray());
    }

    [Fact]
    public void LongSequenceScalarsAndStftStepsAreBounded()
    {
        var input = DenseTensor<float>.OfValues(new[] { 1f, 2f, 3f });
        Assert.ThrowsAny<ArgumentException>(() => CPUExecutionProvider.SplitToSequence(input, Scalar(Bound + 1), 0, 1, null));
        var sequence = new TensorSequence(new[] { input });
        Assert.ThrowsAny<ArgumentException>(() => CPUExecutionProvider.SequenceAt(sequence, Scalar(Bound + 1), null));
        var signal = new DenseTensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 1, 4 });
        Assert.ThrowsAny<ArgumentException>(() => CPUExecutionProvider.STFT(signal, Scalar(Bound + 1), null, Scalar(2), 1, null));
        Assert.Equal(OpStatus.Success, CPUExecutionProvider.STFT(signal, Scalar(Bound), null, Scalar(2), 1, null).Status);
    }

    [Fact]
    public void DescriptorProductsCannotHideOversizedGeometry()
    {
        foreach (var dimensions in new[] { new[] { Bound + 1, 0L }, new[] { -1L, Bound + 1 }, new[] { 1L << 24, (1L << 24) + 1 } })
        {
            var tensor = new TensorProto { DataType = (int)TensorElementType.Float, RawData = ByteString.Empty };
            tensor.Dims.Add(dimensions);
            Assert.ThrowsAny<ArgumentException>(() => tensor.GetTensorData());
        }
        var payload = new TensorProto { DataType = (int)TensorElementType.Int64 };
        payload.Dims.Add(2);
        payload.Int64Data.Add(new[] { long.MinValue, long.MaxValue });
        Assert.Equal(new[] { long.MinValue, long.MaxValue }, (long[])payload.GetTensorData());
    }

    [Fact]
    public void ModelAndImportDescriptorsUseTheSameLimit()
    {
        var dimensions = new[] { 1 << 24, (1 << 24) + 1 };
        var model = new OnnxModel();
        model.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = dimensions });
        Assert.ThrowsAny<ArgumentException>(() => Model.Load(model));
        var value = new ValueInfoProto { Name = "x", Type = new TypeProto { TensorType = new TypeProto.Types.Tensor { ElemType = (int)TensorElementType.Float, Shape = new TensorShapeProto() } } };
        foreach (var dimension in dimensions) value.Type.TensorType.Shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = dimension });
        Assert.ThrowsAny<ArgumentException>(() => value.ToValueDto());
    }
}
