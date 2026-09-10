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
        Assert.Single(((TensorSequence)g.Outputs["s"]).Items);
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

    [Fact]
    public void SequenceAt_NegativeIndex_CountsFromEnd()
    {
        // ORT 1.29: position -1 yields the last part.
        var x = DenseTensor<float>.OfValues(new float[,] { { 0f, 1f, 2f, 3f, 4f, 5f }, { 6f, 7f, 8f, 9f, 10f, 11f } });
        var sp = CPUExecutionProvider.SplitToSequence(x, DenseTensor<int>.OfValues(new int[] { 2, 4 }), 1, 0, null);
        Assert.Equal(OpStatus.Success, sp.Status);
        var idx = DenseTensor<long>.OfShape();
        idx.SetValue(0, -1L);
        var r = CPUExecutionProvider.SequenceAt((TensorSequence)sp.Outputs[0], idx, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 2f, 3f, 4f, 5f, 8f, 9f, 10f, 11f }, ((Tensor<float>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void SequenceAt_OutOfRange_Fails()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 0f, 1f, 2f, 3f, 4f, 5f }, { 6f, 7f, 8f, 9f, 10f, 11f } });
        var sp = CPUExecutionProvider.SplitToSequence(x, DenseTensor<int>.OfValues(new int[] { 2, 4 }), 1, 0, null);
        Assert.Equal(OpStatus.Success, sp.Status);
        var idx = DenseTensor<long>.OfShape();
        idx.SetValue(0, 5L);
        var r = CPUExecutionProvider.SequenceAt((TensorSequence)sp.Outputs[0], idx, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void SplitToSequence_NullSplit_ChunksOnes()
    {
        // ORT 1.29: absent split chunks size 1 ([[1,2,3,4]] on axis 1
        // yields four (1,1) pieces).
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f } });
        var r = CPUExecutionProvider.SplitToSequence(x, null, 1, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var items = ((TensorSequence)r.Outputs![0]).Items;
        Assert.Equal(4, items.Count);
        Assert.Equal(new float[] { 1f }, ((Tensor<float>)items[0]).ToArray());
        Assert.Equal(new float[] { 4f }, ((Tensor<float>)items[3]).ToArray());
    }

    [Fact]
    public void SplitToSequence_NullAxis_DefaultsToZero()
    {
        // ORT 1.29: omitted axis splits axis 0 ([[1,2],[3,4]] with sizes
        // [1,1] yields [(1,2),(1,2)]).
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var r = CPUExecutionProvider.SplitToSequence(x, DenseTensor<long>.OfValues(new long[] { 1L, 1L }), null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var items = ((TensorSequence)r.Outputs![0]).Items;
        Assert.Equal(2, items.Count);
        Assert.Equal(new float[] { 1f, 2f }, ((Tensor<float>)items[0]).ToArray());
        Assert.Equal(new float[] { 3f, 4f }, ((Tensor<float>)items[1]).ToArray());
    }

    [Fact]
    public void SplitToSequence_FloatSplit_FailsCleanly()
    {
        // ORT refuses scalar and vector float splits at load; the
        // provider must fail descriptively instead of throwing from
        // ToInt64Scalar/ToIntArray.
        var x = DenseTensor<float>.OfShape(1, 4);
        var scalar = new DenseTensor<float>(new float[] { 2f }, Array.Empty<int>());
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.SplitToSequence(x, scalar, 1, null, null).Status);
        var vector = DenseTensor<float>.OfValues(new float[] { 1f, 3f });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.SplitToSequence(x, vector, 1, null, null).Status);
    }

    [Fact]
    public void SequenceAt_FloatIndex_FailsCleanly()
    {
        // ORT refuses a non-int SequenceAt index at load; the provider
        // must fail descriptively instead of throwing from ToInt64Scalar.
        var x = DenseTensor<float>.OfValues(new float[,] { { 0f, 1f, 2f, 3f, 4f, 5f }, { 6f, 7f, 8f, 9f, 10f, 11f } });
        var sp = CPUExecutionProvider.SplitToSequence(x, DenseTensor<int>.OfValues(new int[] { 2, 4 }), 1, 0, null);
        Assert.Equal(OpStatus.Success, sp.Status);
        var f = DenseTensor<float>.OfValues(new float[] { 0f });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.SequenceAt((TensorSequence)sp.Outputs[0], f, null).Status);
    }

    [Fact]
    public void SequenceAt_NonSequence_FailsCleanly()
    {
        // ORT refuses a non-sequence SequenceAt input at load; the
        // provider must fail descriptively instead of throwing.
        var t = DenseTensor<float>.OfValues(new float[] { 1f });
        var idx = DenseTensor<long>.OfValues(new long[] { 0L });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.SequenceAt(t, idx, null).Status);
    }

}
