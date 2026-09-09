using System;
using System.Collections.Generic;

namespace Lokad.Onnx.Tensors.Tests;

public class SoftmaxOpsetTests
{
    static float[] Zeros(int d0, int d1, int d2) => new float[d0 * d1 * d2];

    static void AssertAll(float[] actual, float expected)
    {
        foreach (var v in actual) Assert.Equal(expected, v, 5);
    }

    [Fact]
    public void NewOpset_NonLastAxis_NormalizesSingleAxis()
    {
        var x = new DenseTensor<float>(Zeros(1, 2, 2), new int[] { 1, 2, 2 });
        var y = Tensor<float>.Softmax(x, 1, null, 13);
        Assert.Equal(new[] { 1, 2, 2 }, y.Dimensions.ToArray());
        AssertAll(y.ToArray(), 0.5f);
    }

    [Fact]
    public void OldOpset_NonLastAxis_FlattensSuffix()
    {
        var x = new DenseTensor<float>(Zeros(1, 2, 2), new int[] { 1, 2, 2 });
        var y = Tensor<float>.Softmax(x, 1, null, 12);
        Assert.Equal(new[] { 1, 2, 2 }, y.Dimensions.ToArray());
        AssertAll(y.ToArray(), 0.25f);
    }

    [Fact]
    public void NewOpset_DefaultAxis_IsLast()
    {
        var x = new DenseTensor<float>(Zeros(1, 2, 2), new int[] { 1, 2, 2 });
        var y = Tensor<float>.Softmax(x, -1, null, 13);
        AssertAll(y.ToArray(), 0.5f);
    }

    [Fact]
    public void Arithmetic_ChargedToMathStage()
    {
        using var enabled = Profiler.BeginExecution(true);
        var x = new DenseTensor<float>(Zeros(1, 2, 2), new int[] { 1, 2, 2 });
        Profiler.StartNodeProfile(1, OpType.Softmax);
        var y = Tensor<float>.Softmax(x, -1, null, 13);
        Profiler.StopNodeProfile();
        AssertAll(y.ToArray(), 0.5f);
        AssertStages(enabled, OpType.Softmax);
        var destination = new DenseTensor<float>(new int[] { 1, 2, 2 });
        Profiler.StartNodeProfile(2, OpType.Softmax);
        Tensor<float>.Softmax(x, destination, -1, null, 13);
        Profiler.StopNodeProfile();
        AssertAll(destination.ToArray(), 0.5f);
        AssertStages(enabled, OpType.Softmax);
        var xd = new DenseTensor<double>(new double[] { 0.0, 0.0, 0.0, 0.0 }, new int[] { 1, 2, 2 });
        Profiler.StartNodeProfile(3, OpType.Softmax);
        var yd = Tensor<double>.Softmax(xd, -1, null, 13);
        Profiler.StopNodeProfile();
        foreach (var v in yd.ToArray()) Assert.Equal(0.5, v, 5);
        AssertStages(enabled, OpType.Softmax);
    }

    static void AssertStages(ProfilerContext enabled, OpType op)
    {
        var node = enabled.Profile.Peek();
        Assert.Equal(op, node.Op);
        var stages = new List<OpStage>();
        foreach (var profile in node.OpsProfile) stages.Add(profile.Stage);
        Assert.Contains(OpStage.ValidateArguments, stages);
        Assert.Contains(OpStage.Math, stages);
    }

    [Fact]
    public void OldOpset_DefaultAxis_IsOne()
    {
        var graph = new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = 12 },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
        graph.Inputs["x"] = new DenseTensor<float>(Zeros(1, 2, 2), new int[] { 1, 2, 2 });
        var node = new Node
        {
            Name = "sm", Op = OpType.Softmax, Inputs = new[] { "x" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object>(),
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, r.Status);
        AssertAll(((Tensor<float>)r.Outputs[0]).ToArray(), 0.25f);
    }

    [Fact]
    public void NewOpset_DefaultAxis_IsLastThroughGraph()
    {
        var graph = new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = 18 },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
        graph.Inputs["x"] = new DenseTensor<float>(Zeros(1, 2, 2), new int[] { 1, 2, 2 });
        var node = new Node
        {
            Name = "sm", Op = OpType.Softmax, Inputs = new[] { "x" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object>(),
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, r.Status);
        AssertAll(((Tensor<float>)r.Outputs[0]).ToArray(), 0.5f);
    }

    [Fact]
    public void NegativeAxis_MatchesPositiveForm()
    {
        var data = new float[1, 2, 2] { { { 0f, 1f }, { 2f, 3f } } };
        var a = Tensor<float>.Softmax(DenseTensor<float>.OfValues(data), -1, null, 13);
        var b = Tensor<float>.Softmax(DenseTensor<float>.OfValues(data), 2, null, 13);
        var aa = a.ToArray(); var bb = b.ToArray();
        for (int i = 0; i < aa.Length; i++) Assert.Equal(bb[i], aa[i], 6);
        var c = Tensor<float>.Softmax(DenseTensor<float>.OfValues(data), -2, null, 12);
        var d = Tensor<float>.Softmax(DenseTensor<float>.OfValues(data), 1, null, 12);
        var cc = c.ToArray(); var dd = d.ToArray();
        for (int i = 0; i < cc.Length; i++) Assert.Equal(dd[i], cc[i], 6);
    }

    [Fact]
    public void InvalidAxis_Throws()
    {
        var x = new DenseTensor<float>(Zeros(1, 2, 2), new int[] { 1, 2, 2 });
        Assert.Throws<ArgumentException>(() => Tensor<float>.Softmax(x, 3, null, 13));
        Assert.Throws<ArgumentException>(() => Tensor<float>.Softmax(x, -4, null, 13));
    }

    [Fact]
    public void ScalarRank_Throws()
    {
        var s = DenseTensor<float>.Scalar(1f);
        Assert.Throws<ArgumentException>(() => Tensor<float>.Softmax(s, -1, null, 13));
        Assert.Throws<ArgumentException>(() => Tensor<float>.Softmax(s, 0, null, 12));
    }

    [Fact]
    public void ScalarAndAutoOptions_AgreeOnValues()
    {
        var data = new float[2, 3] { { 0f, 1f, 2f }, { 2f, 1f, 0f } };
        var a = Tensor<float>.Softmax(DenseTensor<float>.OfValues(data), 1, TensorExecutionOptions.Scalar, 13);
        var b = Tensor<float>.Softmax(DenseTensor<float>.OfValues(data), 1, TensorExecutionOptions.Auto, 13);
        var aa = a.ToArray(); var bb = b.ToArray();
        for (int i = 0; i < aa.Length; i++) Assert.Equal(bb[i], aa[i], 6);
        var dest1 = DenseTensor<float>.OfShape(2, 3);
        var dest2 = DenseTensor<float>.OfShape(2, 3);
        Tensor<float>.Softmax(DenseTensor<float>.OfValues(data), dest1, 1, TensorExecutionOptions.Scalar, 13);
        Tensor<float>.Softmax(DenseTensor<float>.OfValues(data), dest2, 1, TensorExecutionOptions.Auto, 13);
        var d1 = dest1.ToArray(); var d2 = dest2.ToArray();
        for (int i = 0; i < d1.Length; i++) Assert.Equal(d2[i], d1[i], 6);
    }

    [Fact]
    public void Double_NonLastAxis_FollowsVersionRule()
    {
        var x = new DenseTensor<double>(new double[4], new int[] { 1, 2, 2 });
        var n = Tensor<double>.Softmax(x, 1, null, 13);
        foreach (var v in n.ToArray()) Assert.Equal(0.5, v, 9);
        var o = Tensor<double>.Softmax(x, 1, null, 12);
        foreach (var v in o.ToArray()) Assert.Equal(0.25, v, 9);
    }

    [Fact]
    public void NonLastAxis_ValuesMatchIndependentReference()
    {
        var data = new float[1, 2, 2] { { { 0f, 1f }, { 2f, 3f } } };
        var y = Tensor<float>.Softmax(DenseTensor<float>.OfValues(data), 1, null, 13);
        float e0 = MathF.Exp(0f) / (MathF.Exp(0f) + MathF.Exp(2f));
        float e1 = MathF.Exp(1f) / (MathF.Exp(1f) + MathF.Exp(3f));
        float e2 = MathF.Exp(2f) / (MathF.Exp(0f) + MathF.Exp(2f));
        float e3 = MathF.Exp(3f) / (MathF.Exp(1f) + MathF.Exp(3f));
        var a = y.ToArray();
        Assert.Equal(e0, a[0], 5); Assert.Equal(e1, a[1], 5);
        Assert.Equal(e2, a[2], 5); Assert.Equal(e3, a[3], 5);
    }
}
