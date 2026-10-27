using System;
using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class ResizeIdentityTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (string dtype in new[] { "float32", "float64", "int8", "uint8", "int32" })
        foreach (string mode in new[] { "nearest", "linear", "cubic" })
        foreach (int form in new[] { 0, 1, 2 }) // Unit scales, fractional scales, explicit sizes.
            if (mode != "cubic" || dtype is "float32" or "float64") yield return [dtype, mode, form];
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void UnchangedShapeCopiesLogicalValuesWithoutAliasing(string dtype, string mode, int form)
    {
        for (int layout = 0; layout < 4; layout++) CheckIdentity(Input(dtype, layout), mode, form);
    }

    [Theory]
    [InlineData(false, "nearest")] [InlineData(false, "linear")] [InlineData(false, "cubic")]
    [InlineData(true, "nearest")] [InlineData(true, "linear")] [InlineData(true, "cubic")]
    public void FloatingIdentityPreservesSignedZeroInfinityAndNaNPayload(bool wide, string mode)
    {
        ITensor input = wide
            ? Layout(new[] { -0d, BitConverter.Int64BitsToDouble(0x7ff8123456789abc), double.PositiveInfinity, double.NegativeInfinity }, 1)
            : Layout(new[] { -0f, BitConverter.Int32BitsToSingle(0x7fc12345), float.PositiveInfinity, float.NegativeInfinity }, 1);
        for (int form = 0; form < 3; form++) CheckIdentity(input, mode, form);
    }

    [Theory]
    [InlineData("float32")] [InlineData("float64")] [InlineData("int8")]
    [InlineData("uint8")] [InlineData("int32")]
    public void ChangedShapeStillResizes(string dtype)
    {
        var input = Input(dtype, 0);
        var result = CPUExecutionProvider.Resize(input, null, DenseTensor<float>.OfValues([1, 1, 1, 2]), null,
            "nearest", "asymmetric", "floor", null, null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new[] { 1, 1, 2, 4 }, result.Outputs[0].Dims);
        Assert.Equal(Bits(input).SelectMany(value => new[] { value, value }), Bits(result.Outputs[0]));
    }

    [Theory]
    [InlineData("integer-cubic")] [InlineData("mode")] [InlineData("coordinate")] [InlineData("nearest")]
    [InlineData("zero")] [InlineData("negative")] [InlineData("nan")] [InlineData("infinity")]
    [InlineData("scale-count")] [InlineData("rank")] [InlineData("batch")]
    public void ProviderKeepsRejectingInvalidOrUnsupportedConfigurations(string defect)
    {
        ITensor input = Input(defect == "integer-cubic" ? "int32" : "float32", 0);
        if (defect == "rank") input = DenseTensor<float>.OfShape(1, 2, 2);
        float scale = defect switch { "zero" => 0, "negative" => -1, "nan" => float.NaN, "infinity" => float.PositiveInfinity, _ => 1.25f };
        var scales = DenseTensor<float>.OfValues(defect == "scale-count" ? [1, 1, scale] : new[] { 1, 1, 1, scale });
        var sizes = defect == "batch" ? DenseTensor<long>.OfValues([2, 1, 2, 2]) : null;
        var result = CPUExecutionProvider.Resize(input, null, sizes is null ? scales : null, sizes,
            defect == "integer-cubic" ? "cubic" : defect == "mode" ? "unsupported" : "nearest",
            defect == "coordinate" ? "unsupported" : "asymmetric", defect == "nearest" ? "unsupported" : "floor", null, null, null);
        Assert.Equal(OpStatus.Failure, result.Status);
        Assert.Empty(result.Outputs);
    }

    [Theory]
    [InlineData("mode")] [InlineData("coordinate")] [InlineData("nearest")] [InlineData("integer-cubic")]
    public void TypedIdentityDoesNotBypassUnsupportedModes(string defect)
    {
        var mode = defect == "mode" ? (MathOps.ResizeMode)999 : defect == "integer-cubic" ? MathOps.ResizeMode.Cubic : MathOps.ResizeMode.Nearest;
        var coordinate = defect == "coordinate" ? (MathOps.ResizeCoordinateTransformation)999 : MathOps.ResizeCoordinateTransformation.Asymmetric;
        var nearest = defect == "nearest" ? (MathOps.ResizeNearestMode)999 : MathOps.ResizeNearestMode.Floor;
        Assert.Throws<NotSupportedException>(() =>
        {
            if (defect == "integer-cubic") Tensor<int>.Resize((Tensor<int>)Input("int32", 0), [1, 1, 2, 2], mode, coordinate, nearest, -.75f, [1, 1, 1, 1.25]);
            else Tensor<float>.Resize((Tensor<float>)Input("float32", 0), [1, 1, 2, 2], mode, coordinate, nearest, -.75f, [1, 1, 1, 1.25]);
        });
    }

    static void CheckIdentity(ITensor input, string mode, int form)
    {
        var expected = Bits(input);
        double[]? trueScales = form == 2 ? null : [1, 1, 1, form == 0 ? 1 : 1.25];
        ITensor argument = form == 2 ? DenseTensor<long>.OfValues([1, 1, 2, 2])
            : DenseTensor<float>.OfValues(trueScales!.Select(value => (float)value).ToArray());
        var originalArgument = Bits(argument);
        var resizeMode = mode == "nearest" ? MathOps.ResizeMode.Nearest : mode == "linear" ? MathOps.ResizeMode.Linear : MathOps.ResizeMode.Cubic;
        const MathOps.ResizeCoordinateTransformation coordinate = MathOps.ResizeCoordinateTransformation.Asymmetric;
        const MathOps.ResizeNearestMode nearest = MathOps.ResizeNearestMode.Floor;
        ITensor typed = input switch
        {
            Tensor<float> x => Tensor<float>.Resize(x, [1, 1, 2, 2], resizeMode, coordinate, nearest, -.75f, trueScales),
            Tensor<double> x => Tensor<double>.Resize(x, [1, 1, 2, 2], resizeMode, coordinate, nearest, -.75, trueScales),
            Tensor<sbyte> x => Tensor<sbyte>.Resize(x, [1, 1, 2, 2], resizeMode, coordinate, nearest, -.75f, trueScales),
            Tensor<byte> x => Tensor<byte>.Resize(x, [1, 1, 2, 2], resizeMode, coordinate, nearest, -.75f, trueScales),
            Tensor<int> x => Tensor<int>.Resize(x, [1, 1, 2, 2], resizeMode, coordinate, nearest, -.75f, trueScales),
            _ => throw new InvalidOperationException()
        };
        CheckOutput(typed);
        var model = new OnnxModel
        {
            Opset = { [""] = 13 },
            Inputs = { new OnnxValueInfo { Name = "x", ElementType = input.ElementType, Dims = [1, 1, 2, 2] } },
            Outputs = { new OnnxValueInfo { Name = "y", ElementType = input.ElementType, Dims = [1, 1, 2, 2] } },
            Initializers = { new OnnxTensor { Name = "arg", ElementType = argument.ElementType, Dims = [4], Data = argument.ToArray() } },
            Nodes = { new OnnxNode { Name = "resize", OpType = "Resize", Inputs = form == 2 ? ["x", "", "", "arg"] : ["x", "", "arg"],
                Outputs = ["y"], Attributes = new() { ["mode"] = mode, ["coordinate_transformation_mode"] = "asymmetric", ["nearest_mode"] = "floor" } } }
        };
        var graph = Model.Load(model);
        foreach (var options in new[] { ExecutionOptions.Default, ExecutionOptions.Scalar, ExecutionOptions.Simd })
        {
            var result = CPUExecutionProvider.Resize(input, null, form == 2 ? null : argument, form == 2 ? argument : null,
                mode, "asymmetric", "floor", null, null, options);
            Assert.Equal(OpStatus.Success, result.Status);
            CheckOutput(result.Outputs[0]);
            Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true, ExecutionProvider.CPU, options), graph.LastErrorMessage);
            CheckOutput(graph.Outputs["y"]);
            Assert.Equal(originalArgument, Bits(argument));
        }

        void CheckOutput(ITensor output)
        {
            Assert.Equal(input.ElementType, output.ElementType);
            Assert.Equal(new[] { 1, 1, 2, 2 }, output.Dims);
            Assert.Equal(expected, Bits(output));
            Assert.Equal(expected, Bits(input));
            Assert.NotSame(input, output);
            output.SetValue(0, output.GetValue(1));
            Assert.Equal(expected, Bits(input));
        }
    }

    static ITensor Input(string dtype, int layout) => dtype switch
    {
        "float32" => Layout(new float[] { 1, 10, 30, 100 }, layout),
        "float64" => Layout(new double[] { 1, 10, 30, 100 }, layout),
        "int8" => Layout(new sbyte[] { -128, -1, 64, 127 }, layout),
        "uint8" => Layout(new byte[] { 0, 1, 254, 255 }, layout),
        "int32" => Layout(new[] { int.MinValue, int.MaxValue, -16777217, 16777217 }, layout),
        _ => throw new InvalidOperationException()
    };

    static Tensor<T> Layout<T>(T[] values, int layout) where T : unmanaged
    {
        Tensor<T> input = layout == 2
            ? DenseTensor<T>.OfShape(1, 1, 2, 3).Slice(SliceIndex.All, SliceIndex.All, SliceIndex.All, new SliceIndex(0, 2))
            : new DenseTensor<T>(new[] { 1, 1, layout == 3 ? 1 : 2, 2 }, layout == 1);
        for (int r = 0; r < (layout == 3 ? 1 : 2); r++)
        for (int c = 0; c < 2; c++) input[0, 0, r, c] = values[2 * r + c];
        return layout == 3 ? input.BroadcastDim(2, 2) : input;
    }

    static long[] Bits(ITensor tensor) => tensor switch
    {
        Tensor<float> x => x.ToArray().Select(value => (long)BitConverter.SingleToInt32Bits(value)).ToArray(),
        Tensor<double> x => x.ToArray().Select(BitConverter.DoubleToInt64Bits).ToArray(),
        _ => tensor.ToArray().Cast<object>().Select(Convert.ToInt64).ToArray()
    };
}
