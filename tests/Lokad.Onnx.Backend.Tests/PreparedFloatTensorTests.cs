namespace Lokad.Onnx.Backend.Tests;

using System;
using System.Linq;
using System.Threading.Tasks;
using Xunit;

public class PreparedFloatTensorTests
{
    static PreparedFloatValue Prepare(params float[] values) => new(new DenseTensor<float>(values, new[] { 2, values.Length / 2 }));
    static int[] Bits(Tensor<float> value) => value.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray();

    [Fact]
    public void IndependentUsesAndOriginalSourceKeepEveryBit()
    {
        float[] original = { 0f, -0f, float.PositiveInfinity, BitConverter.Int32BitsToSingle(unchecked((int)0xffc12345)) };
        int[] expected = original.Select(BitConverter.SingleToInt32Bits).ToArray();
        var prepared = Prepare(original);
        original[0] = 8f;
        var a = prepared.CreateTensor(); var b = prepared.CreateTensor();
        Assert.Equal(expected, Bits(a));
        Assert.False(a.HasMaterialized);
        Assert.False(b.HasMaterialized);
        a.SetValue(0, 19f);
        Assert.True(a.HasMaterialized);
        Assert.Equal(19f, a.GetValue(0));
        Assert.Equal(expected, Bits(b));
        Assert.Equal(expected, Bits(prepared.CreateTensor()));
    }

    [Fact]
    public void PublicMemoryDenseViewsReshapesSlicesAndClonesAreSafe()
    {
        var prepared = Prepare(1f, 2f, 3f, 4f, 5f, 6f);
        var a = prepared.CreateTensor();
        var reshaped = a.Reshape(new[] { 3, 2 });
        var slice = a.Slice(SliceIndex.All, new SliceIndex("1:"));
        reshaped[2, 1] = 12f;
        Assert.Equal(12f, a[1, 2]);
        slice[0, 0] = 17f;
        Assert.Equal(17f, a[0, 1]);
        var dense = a.ToDenseTensor();
        dense.Buffer.Span[0] = 21f;
        Assert.Equal(21f, reshaped[0, 0]);
        a.Storage.Span[2] = 22f;
        Assert.Equal(22f, a[0, 2]);
        var clone = a.Clone(); clone.SetValue(0, 99f);
        var array = a.ToArray(); array[0] = 98f;
        Assert.Equal(21f, a.GetValue(0));
        Assert.Equal(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, prepared.CreateTensor().ToArray());
        Assert.Throws<ArgumentException>(() => a.Reshape(new[] { 3, 3 }));
    }

    [Fact]
    public void StorageAndSliceOffsetsMatchDenseTensor()
    {
        var a = Prepare(1f, 2f, 3f, 4f, 5f, 6f).CreateTensor();
        Assert.Equal(5, a.GetStorageIndex(new[] { 1, 2 }));
        var slice = a.Slice(new SliceIndex("1:"), SliceIndex.All);
        Assert.Equal(5, slice.GetStorageIndex(new[] { 0, 2 }));
        slice.Storage.Span[slice.GetStorageIndex(new[] { 0, 2 })] = 33f;
        Assert.Equal(33f, a.GetValue(5));
        Assert.True(TensorAlias.SharesBackingMemory(a, slice));
        Assert.False(TensorAlias.SharesBackingMemory(a, a.Clone()));
    }

    [Fact]
    public void BorrowedWeightsStayUnmaterializedAndResultsAreIndependent()
    {
        var prepared = Prepare(1f, 2f, 3f, 4f, 5f, 6f);
        var a = prepared.CreateTensor();
        var x = new DenseTensor<float>(new[] { 2f, 3f }, new[] { 1, 2 });
        var result = CPUExecutionProvider.MatMul(x, a.BorrowReadOnly(), ExecutionOptions.Default, null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new[] { 14f, 19f, 24f }, ((Tensor<float>)result.Outputs[0]).ToArray());
        Assert.False(a.HasMaterialized);
        result.Outputs[0].SetValue(0, 100f);
        Assert.Equal(1f, a.GetValue(0));
        a.SetValue(0, 11f);
        var next = CPUExecutionProvider.MatMul(x, a.BorrowReadOnly(), ExecutionOptions.Default, null);
        Assert.Equal(34f, (float)next.Outputs[0].GetValue(0));
        Assert.Equal(1f, prepared.CreateTensor().GetValue(0));
    }

    [Fact]
    public void ScalarAndEmptyShapesKeepDenseSemantics()
    {
        var scalar = new PreparedFloatValue(new DenseTensor<float>(new[] { -0f }, Array.Empty<int>())).CreateTensor();
        Assert.Equal(unchecked((int)0x80000000), BitConverter.SingleToInt32Bits(scalar[ReadOnlySpan<int>.Empty]));
        var vector = scalar.Reshape(new[] { 1 }); vector.SetValue(0, 7f);
        Assert.Equal(7f, scalar.GetValue(0));
        var empty = new PreparedFloatValue(new DenseTensor<float>(new[] { 0, 3 })).CreateTensor();
        Assert.Empty(empty.ToArray());
        Assert.Empty(empty.Reshape(new[] { 3, 0 }).ToArray());
        Assert.Empty(empty.Storage.ToArray());
        Assert.Throws<IndexOutOfRangeException>(() => empty.GetValue(0));
        Assert.Throws<ArgumentException>(() => empty.Reshape(new[] { 1 }));
    }

    [Fact]
    public void MaterializationCopiesAreCountedOnceAndPreparedReadsStayPrivate()
    {
        var prepared = Prepare(1f, 2f, 3f, 4f);
        var reporter = new CopyAccountant();
        var value = prepared.CreateTensor(reporter);
        Assert.Equal(1f, value.GetValue(0));
        Assert.Equal(0, reporter.TotalCopyBytes);
        value.Reshape(new[] { 4 }).SetValue(0, 8f);
        Assert.Equal(16, reporter.TotalCopyBytes);
        value.Storage.Span[1] = 9f;
        Assert.Equal(16, reporter.TotalCopyBytes);
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, prepared.CreateTensor().ToArray());
    }

    [Fact]
    public void NamedViewsPreserveMetadataAndIndependentUses()
    {
        var prepared = Prepare(1f, 2f, 3f, 4f);
        var value = prepared.CreateTensor(); value.Name = "matrix_weights";
        Assert.Equal("matrix_weights", value.BorrowReadOnly().Name);
        Assert.False(value.HasMaterialized);
        var dense = value.ToDenseTensor();
        Assert.Equal("matrix_weights", dense.Name);
        dense.SetValue(0, 17f);
        Assert.Equal(17f, value.GetValue(0));
        Assert.Equal(1f, prepared.CreateTensor().GetValue(0));
    }

    [Fact]
    public void SeparateContextsOwnTheirWritesAndHeldViews()
    {
        var prepared = Prepare(1f, 2f, 3f, 4f);
        var held = prepared.CreateTensor();
        Parallel.For(0, 64, index =>
        {
            var value = prepared.CreateTensor();
            var view = value.Reshape(new[] { 4 });
            view.SetValue(0, index + 100f);
            Assert.Equal(index + 100f, value.GetValue(0));
        });
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, held.ToArray());
        Assert.False(held.HasMaterialized);
        var alias = held.Reshape(new[] { 4 });
        Parallel.For(0, 4, index => alias.SetValue(index, index + 20f));
        Assert.Equal(new[] { 20f, 21f, 22f, 23f }, held.ToArray());
    }
}
