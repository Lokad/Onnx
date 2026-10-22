using System.Numerics;
using System.Runtime.InteropServices;

namespace Lokad.Onnx.Backend.Tests;

public class ConstantDeduplicationBitTests
{
    [Theory]
    [InlineData("half", 0x0000, 0x8000)]
    [InlineData("half", 0x7e01, 0x7e02)]
    [InlineData("bfloat", 0x0000, 0x8000)]
    [InlineData("bfloat", 0x7fc1, 0x7fc2)]
    public void DifferentSmallFloatBitsDoNotMerge(string type, int left, int right)
    {
        if (type == "half")
            CheckPair(Scalar(BitConverter.UInt16BitsToHalf((ushort)left)), Scalar(BitConverter.UInt16BitsToHalf((ushort)right)), false);
        else CheckPair(Scalar(new BFloat16((ushort)left)), Scalar(new BFloat16((ushort)right)), false);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void WiderFloatBitsDoNotMerge(bool nan)
    {
        CheckPair(Scalar(nan ? BitConverter.Int32BitsToSingle(0x7fc00001) : 0f),
            Scalar(nan ? BitConverter.Int32BitsToSingle(0x7fc00002) : -0f), false);
        CheckPair(Scalar(nan ? BitConverter.Int64BitsToDouble(0x7ff8000000000001) : 0d),
            Scalar(nan ? BitConverter.Int64BitsToDouble(0x7ff8000000000002) : -0d), false);
        CheckPair(Scalar(new Complex(1d, nan ? BitConverter.Int64BitsToDouble(0x7ff8000000000001) : 0d)),
            Scalar(new Complex(1d, nan ? BitConverter.Int64BitsToDouble(0x7ff8000000000002) : -0d)), false);
    }

    [Fact]
    public void IdenticalBitsMergeAcrossEveryMappedStorageType()
    {
        CheckSame(true); CheckSame((sbyte)-3); CheckSame((byte)3);
        CheckSame((short)-300); CheckSame((ushort)300); CheckSame(-70000); CheckSame(70000u);
        CheckSame(long.MinValue); CheckSame(ulong.MaxValue);
        CheckSame(BitConverter.UInt16BitsToHalf(0xfe01)); CheckSame(new BFloat16(0xffc1));
        CheckSame(BitConverter.Int32BitsToSingle(unchecked((int)0xffc00001)));
        CheckSame(BitConverter.Int64BitsToDouble(unchecked((long)0xfff8000000000001)));
        CheckSame(new Complex(-0d, BitConverter.Int64BitsToDouble(0x7ff8000000000001)));
    }

    [Fact]
    public void ComparisonUsesLogicalOrderAcrossReversedAndSlicedStorage()
    {
        var values = new Half[] { (Half)1, (Half)2, (Half)3, (Half)4 };
        var rowMajor = new DenseTensor<Half>(values, new[] { 2, 2 });
        var reversed = new DenseTensor<Half>(new Half[] { (Half)1, (Half)3, (Half)2, (Half)4 }, new[] { 2, 2 }, reverseStride: true);
        CheckPair(rowMajor, reversed, true);
        var root = new DenseTensor<Half>(new Half[] { (Half)9, (Half)9, (Half)1, (Half)2, (Half)3, (Half)4 }, new[] { 3, 2 });
        CheckPair(rowMajor, root.Slice(new SliceIndex(1, 3), SliceIndex.All), true);
    }

    static DenseTensor<T> Scalar<T>(T value) where T : unmanaged => DenseTensor<T>.OfValues(new[] { value });

    static void CheckSame<T>(T value) where T : unmanaged => CheckPair(Scalar(value), Scalar(value), true);

    static void CheckPair<T>(Tensor<T> left, Tensor<T> right, bool equal) where T : unmanaged
    {
        var expectedLeft = MemoryMarshal.AsBytes(left.ToArray().AsSpan()).ToArray();
        var expectedRight = MemoryMarshal.AsBytes(right.ToArray().AsSpan()).ToArray();
        var model = new OnnxModel { Name = "constant-bits" };
        model.Opset[""] = 17;
        foreach (var (name, tensor) in new[] { ("left", left), ("right", right) })
        {
            model.Nodes.Add(new OnnxNode { OpType = "Constant", Outputs = [name], Attributes = new() { ["value"] = tensor } });
            model.Nodes.Add(new OnnxNode { OpType = "Identity", Inputs = [name], Outputs = [name + "Out"] });
            model.Outputs.Add(new OnnxValueInfo { Name = name + "Out", ElementType = tensor.ElementType, Dims = tensor.Dimensions.ToArray() });
        }
        var graph = Model.Load(model, maximumPackedWeightBytes: 0);
        Assert.Equal(equal ? 1 : 2, graph.Nodes.Count(n => n.Op == OpType.Constant));
        Assert.True(graph.Execute(new Dictionary<string, ITensor>(), true), graph.LastErrorMessage);
        Assert.Equal(expectedLeft, MemoryMarshal.AsBytes(((Tensor<T>)graph.Outputs["leftOut"]).ToArray().AsSpan()).ToArray());
        Assert.Equal(expectedRight, MemoryMarshal.AsBytes(((Tensor<T>)graph.Outputs["rightOut"]).ToArray().AsSpan()).ToArray());
    }
}
