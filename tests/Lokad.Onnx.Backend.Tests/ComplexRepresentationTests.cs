namespace Lokad.Onnx.Backend.Tests;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using global::Onnx;
using CPU = CPUExecutionProvider;

public class ComplexRepresentationTests
{
    static Complex[] Values() => [new(1e100, 1e-300), new(-0.0, double.PositiveInfinity),
        new(BitConverter.Int64BitsToDouble(0x7ff8000000001234), -0.0), new(double.NegativeInfinity, -1e100)];

    static byte[] Bits(Complex[] values) => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();

    [Fact]
    public void ManagedComplexTypeAndAllWidthApisAgree()
    {
        var value = DenseTensor<Complex>.OfValues(Values());
        Assert.Equal(TensorElementType.Complex128, value.ElementType);
        Assert.Equal(16, Unsafe.SizeOf<Complex>());
        Assert.Equal(16, TensorBase.GetTypeInfo(typeof(Complex))!.TypeSize);
        Assert.Equal(16, TensorBase.GetElementTypeInfo(value.ElementType)!.TypeSize);
        Assert.Equal(16, TensorBase.ElementByteSize(value.ElementType));
        Assert.Null(TensorBase.GetElementTypeInfo(TensorElementType.Complex64));
        Assert.Equal(8, TensorBase.ElementByteSize(TensorElementType.Complex64));
    }

    [Fact]
    public void CorrectlyTypedDtoPreservesDoubleComponentBits()
    {
        var values = Values();
        var tensor = Model.ToTensor(new OnnxTensor { Name = "complex", ElementType = TensorElementType.Complex128, Dims = [2, 2], Data = values });
        Assert.Equal(TensorElementType.Complex128, tensor.ElementType);
        Assert.Equal(Bits(values), Bits(((Tensor<Complex>)tensor).ToArray()));
        Assert.Equal(1e100, ((Tensor<Complex>)tensor).GetValue(0).Real);
    }

    [Fact]
    public void Complex64DtoCannotSilentlyUseDoubleComponents()
    {
        var failure = Assert.Throws<ArgumentException>(() => Model.ToTensor(new OnnxTensor
            { ElementType = TensorElementType.Complex64, Dims = [1], Data = new[] { new Complex(1e100, 2) } }));
        Assert.Contains("Complex64", failure.Message);
    }

    [Fact]
    public void DenseArrayFactoryMatchesSupportedComplexRepresentation()
    {
        Assert.Equal(2, Assert.IsType<Complex[]>(TensorBase.CreateElementArray(TensorElementType.Complex128, 2)).Length);
        Assert.Throws<NotSupportedException>(() => TensorBase.CreateElementArray(TensorElementType.Complex64, 2));
    }

    [Theory]
    [InlineData("reshape")] [InlineData("transpose")] [InlineData("concat")]
    [InlineData("gather")] [InlineData("slice")]
    public void DataMovementPreservesAllDoubleBits(string operation)
    {
        var values = Values();
        var input = DenseTensor<Complex>.OfValues(values, new[] { 2, 2 });
        OpResult result = operation switch
        {
            "reshape" => CPU.Reshape(input, DenseTensor<long>.OfValues([4]), false, null),
            "transpose" => CPU.Transpose(input, [1, 0], null, null),
            "concat" => CPU.Concat([input, input], 0, null),
            "gather" => CPU.Gather(input, DenseTensor<int>.OfValues([1, 0]), 0, null),
            _ => CPU.Slice(input, DenseTensor<int>.OfValues([0, 1]), DenseTensor<int>.OfValues([2, 2]), null, null, null)
        };
        Complex[] expected = operation switch
        {
            "transpose" => [values[0], values[2], values[1], values[3]],
            "concat" => [.. values, .. values],
            "gather" => [values[2], values[3], values[0], values[1]],
            "slice" => [values[1], values[3]],
            _ => values
        };
        Assert.Equal(OpStatus.Success, result.Status);
        var output = Assert.IsAssignableFrom<Tensor<Complex>>(Assert.Single(result.Outputs));
        Assert.Equal(TensorElementType.Complex128, output.ElementType);
        Assert.Equal(Bits(expected), Bits(output.ToArray()));
        Assert.Equal(Bits(values), Bits(input.ToArray()));
    }

    [Fact]
    public void GraphMetricsUseActualComplexPayloadWidth()
    {
        OnnxValueInfo Desc(string name) => new() { Name = name, ElementType = TensorElementType.Complex128, Dims = [4] };
        var graph = Model.Load(new OnnxModel { Name = "complex-payload", Opset = { [""] = 14 },
            Inputs = { Desc("x") }, Outputs = { Desc("y") },
            Nodes = { new OnnxNode { Name = "copy", OpType = "Identity", Inputs = ["x"], Outputs = ["y"] } } });
        var input = DenseTensor<Complex>.OfValues(Values());
        Assert.True(graph.Execute(new ITensor[] { input }, false), graph.LastErrorMessage);
        Assert.Equal(128, graph.LastPeakLiveBytes);
        Assert.Equal(Bits(Values()), Bits(((Tensor<Complex>)graph.Outputs["y"]).ToArray()));
    }

    [Theory]
    [InlineData(TensorElementType.Complex64, false)]
    [InlineData(TensorElementType.Complex128, true)]
    public void SequenceDescriptorsFollowManagedElementSupport(TensorElementType type, bool supported)
    {
        var value = new ValueInfoProto { Name = "values", Type = new TypeProto { SequenceType = new TypeProto.Types.Sequence
            { ElemType = new TypeProto { TensorType = new TypeProto.Types.Tensor { ElemType = (int)type } } } } };
        if (supported) Assert.Equal(type, value.ToValueDto().SequenceElementType);
        else Assert.Throws<NotSupportedException>(() => value.ToValueDto());
    }

    [Theory]
    [InlineData(TensorElementType.Complex64)] [InlineData(TensorElementType.Complex128)]
    public void ExternalComplexPayloadDecodingRemainsExplicitlyUnsupported(TensorElementType type)
    {
        string directory = Path.Combine(Path.GetTempPath(), "lonnx-complex-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            File.WriteAllBytes(Path.Combine(directory, "weights.bin"), new byte[TensorBase.ElementByteSize(type)]);
            var tensor = new TensorProto { Name = "complex-weight", DataType = (int)type, Dims = { 1 },
                DataLocation = TensorProto.Types.DataLocation.External,
                ExternalData = { new StringStringEntryProto { Key = "location", Value = "weights.bin" } } };
            var failure = Assert.Throws<NotSupportedException>(() => tensor.ReadExternalTensorData(directory));
            Assert.Contains(type.ToString(), failure.Message);
        }
        finally
        {
            File.Delete(Path.Combine(directory, "weights.bin"));
            Directory.Delete(directory);
        }
    }
}
