extern alias OnnxSharp;

using OnnxSharp::Onnx;

namespace Lokad.Onnx.Backend.Tests;

public class SequenceValueTests
{
    static OnnxValueInfo Vp(string name, TensorElementType t, int[] dims)
    {
        return new OnnxValueInfo { Name = name, ElementType = t, Dims = dims };
    }

    [Fact]
    public void Import_Accepts_SequenceType()
    {
        var vp = new ValueInfoProto { Name = "s" };
        vp.Type = new TypeProto { SequenceType = new TypeProto.Types.Sequence() };
        var dto = vp.ToValueDto();
        Assert.Equal("s", dto.Name);
        Assert.Equal(TensorElementType.Sequence, dto.ElementType);
        Assert.Empty(dto.Dims);
    }

    [Fact]
    public void SequenceInput_Validates_BySequenceNess()
    {
        var m = new OnnxModel { Name = "m" };
        m.Inputs.Add(Vp("s", TensorElementType.Sequence, new int[0]));
        m.Outputs.Add(Vp("s", TensorElementType.Sequence, new int[0]));
        var g = Model.Load(m);
        var seq = new TensorSequence(new ITensor[] { DenseTensor<float>.OfValues(new float[] { 1f, 2f }) });
        var good = new Dictionary<string, ITensor> { { "s", seq } };
        Assert.True(g.Execute(good, false));
        Assert.Equal(1, ((TensorSequence)g.Outputs["s"]).Items.Count);
        var bad = new Dictionary<string, ITensor> { { "s", DenseTensor<float>.OfValues(new float[] { 1f, 2f }) } };
        Assert.False(g.Execute(bad, false));
    }

    [Fact]
    public void SplitToSequence_Then_SequenceAt_EndToEnd()
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        g.Opset[""] = 11;
        g.Inputs["x"] = DenseTensor<float>.OfShape(1, 4);
        g.Initializers["s"] = DenseTensor<long>.OfValues(new long[] { 2, 2 });
        var idx = DenseTensor<long>.OfShape();
        idx.SetValue(0, 1L);
        g.Initializers["i"] = idx;
        g.Outputs["z"] = DenseTensor<float>.OfShape(1, 2);
        g.Nodes.Add(new Node { Name = "sp", Op = OpType.SplitToSequence, Inputs = new[] { "x", "s" }, Outputs = new[] { "q" }, Attributes = new Dictionary<string, object> { { "axis", 1 } } });
        g.Nodes.Add(new Node { Name = "at", Op = OpType.SequenceAt, Inputs = new[] { "q", "i" }, Outputs = new[] { "z" } });
        g.IntermediateOutputs["q"] = null;
        g.RefreshLifetimeAnalysis();
        var user = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[,] { { 0f, 1f, 2f, 3f } }) } };
        Assert.True(g.Execute(user, true));
        Assert.Equal(new float[] { 2f, 3f }, ((Tensor<float>)g.Outputs["z"]).ToArray());
    }
}
