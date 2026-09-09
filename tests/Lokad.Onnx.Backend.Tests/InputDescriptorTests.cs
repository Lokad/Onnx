extern alias OnnxSharp;

using OnnxSharp::Onnx;

namespace Lokad.Onnx.Backend.Tests;

public class InputDescriptorTests
{
    static OnnxModel EmptyModel(OnnxValueInfo input, OnnxValueInfo output)
    {
        var m = new OnnxModel { Name = "m" };
        m.Inputs.Add(input);
        m.Outputs.Add(output);
        return m;
    }

    static OnnxValueInfo V(string name, int[] dims)
    {
        return new OnnxValueInfo { Name = name, ElementType = TensorElementType.Float, Dims = dims };
    }

    [Fact]
    public void MetadataLoad_Allocates_ByRank_NotElements()
    {
        var warm = EmptyModel(V("x", new int[] { 1, 2 }), V("y", new int[] { 1, 2 }));
        Model.Load(warm).Reset();
        var big = EmptyModel(V("x", new int[] { 1, 10000000 }), V("y", new int[] { 1, 10000000 }));
        long before = GC.GetAllocatedBytesForCurrentThread();
        var g = Model.Load(big);
        g.Reset();
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(allocated < 4000000L, $"metadata load allocated {allocated} bytes for shape-only descriptors");
    }


    static OnnxValueInfo Vp(string name, int[] dims, string?[]? ps)
    {
        return new OnnxValueInfo { Name = name, ElementType = TensorElementType.Float, Dims = dims, DimParams = ps };
    }

    static OnnxModel DescribedModel(OnnxValueInfo input, OnnxValueInfo output)
    {
        var m = new OnnxModel { Name = "m" };
        m.Inputs.Add(input);
        m.Outputs.Add(output);
        return m;
    }

    [Fact]
    public void RealZeroExtent_RejectsNonzero()
    {
        var g = Model.Load(DescribedModel(Vp("x", new int[] { 0 }, new string?[] { null }), Vp("x", new int[] { 0 }, new string?[] { null })));
        var empty = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[0]) } };
        Assert.True(g.Execute(empty, false));
        Assert.Equal(0, ((Tensor<float>)g.Outputs["x"]).Length);
        var bad = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { 1f }) } };
        Assert.False(g.Execute(bad, false));
    }

    [Fact]
    public void SymbolicExtent_AcceptsAny_ArrayForm()
    {
        var g = Model.Load(DescribedModel(Vp("x", new int[] { 0, 3 }, new string?[] { "n", null }), Vp("x", new int[] { 0, 3 }, new string?[] { "n", null })));
        foreach (int n in new int[] { 0, 1, 5 })
        {
            var data = new float[n, 3];
            var user = new ITensor[] { DenseTensor<float>.OfValues(data) };
            Assert.True(g.Execute(user, false));
            Assert.Equal(new int[] { n, 3 }, ((Tensor<float>)g.Outputs["x"]).Dimensions.ToArray());
            g.Reset();
        }
    }

    [Fact]
    public void SharedSymbolic_AgreesAcrossInputs()
    {
        var m = new OnnxModel { Name = "m" };
        m.Inputs.Add(Vp("a", new int[] { 0, 2 }, new string?[] { "n", null }));
        m.Inputs.Add(Vp("b", new int[] { 0, 3 }, new string?[] { "n", null }));
        m.Outputs.Add(Vp("a", new int[] { 0, 2 }, new string?[] { "n", null }));
        m.Outputs.Add(Vp("b", new int[] { 0, 3 }, new string?[] { "n", null }));
        var g = Model.Load(m);
        var agree = new Dictionary<string, ITensor>
        {
            { "a", DenseTensor<float>.OfValues(new float[2, 2] { { 1f, 2f }, { 3f, 4f } }) },
            { "b", DenseTensor<float>.OfValues(new float[2, 3] { { 1f, 2f, 3f }, { 4f, 5f, 6f } }) },
        };
        Assert.True(g.Execute(agree, false));
        Assert.Equal(new int[] { 2, 2 }, ((Tensor<float>)g.Outputs["a"]).Dimensions.ToArray());
        var clash = new Dictionary<string, ITensor>
        {
            { "a", DenseTensor<float>.OfValues(new float[2, 2] { { 1f, 2f }, { 3f, 4f } }) },
            { "b", DenseTensor<float>.OfValues(new float[3, 3] { { 1f, 2f, 3f }, { 4f, 5f, 6f }, { 7f, 8f, 9f } }) },
        };
        Assert.False(g.Execute(clash, false));
    }

    [Fact]
    public void Import_Retains_DimParams()
    {
        var vp = new ValueInfoProto { Name = "x" };
        vp.Type = new TypeProto { TensorType = new TypeProto.Types.Tensor { ElemType = (int)TensorElementType.Float } };
        vp.Type.TensorType.Shape = new TensorShapeProto();
        vp.Type.TensorType.Shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimParam = "batch" });
        vp.Type.TensorType.Shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = 3 });
        var dto = vp.ToValueDto();
        Assert.Equal(new int[] { 0, 3 }, dto.Dims);
        Assert.NotNull(dto.DimParams);
        Assert.Equal(new string?[] { "batch", null }, dto.DimParams);
    }

    [Fact]
    public void Import_Marks_AnonymousUnknown_DistinctFromZero()
    {
        var vp = new ValueInfoProto { Name = "x" };
        vp.Type = new TypeProto { TensorType = new TypeProto.Types.Tensor { ElemType = (int)TensorElementType.Float } };
        vp.Type.TensorType.Shape = new TensorShapeProto();
        vp.Type.TensorType.Shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = 5 });
        vp.Type.TensorType.Shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimParam = "batch" });
        vp.Type.TensorType.Shape.Dim.Add(new TensorShapeProto.Types.Dimension());
        vp.Type.TensorType.Shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = 0 });
        var dto = vp.ToValueDto();
        Assert.Equal(new int[] { 5, 0, -1, 0 }, dto.Dims);
        Assert.NotNull(dto.DimParams);
        Assert.Equal(new string?[] { null, "batch", null, null }, dto.DimParams);
    }

    [Fact]
    public void AnonymousUnknown_AcceptsDifferentExtents()
    {
        var par = new string?[] { null, null };
        var g = Model.Load(DescribedModel(Vp("x", new int[] { -1, 3 }, par), Vp("x", new int[] { -1, 3 }, par)));
        foreach (int n in new int[] { 0, 1, 5 })
        {
            var user = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[n, 3]) } };
            Assert.True(g.Execute(user, false), g.LastErrorMessage);
            Assert.Equal(new int[] { n, 3 }, ((Tensor<float>)g.Outputs["x"]).Dimensions.ToArray());
            g.Reset();
        }
    }

    [Fact]
    public void AnonymousUnknown_DoesNotCoupleDims()
    {
        var par = new string?[] { null, null };
        var g = Model.Load(DescribedModel(Vp("x", new int[] { -1, -1 }, par), Vp("x", new int[] { -1, -1 }, par)));
        foreach (var shape in new int[][] { new int[] { 2, 3 }, new int[] { 4, 1 } })
        {
            var user = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfShape(shape) } };
            Assert.True(g.Execute(user, false), g.LastErrorMessage);
            g.Reset();
        }
    }
}
