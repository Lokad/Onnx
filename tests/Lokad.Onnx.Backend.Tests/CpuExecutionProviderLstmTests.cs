using System;
using System.Linq;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

// Differential LSTM coverage (PLAN.md Milestone 3). Expected values are
// frozen Python onnxruntime 1.29.0 oracles (ORT_SEQUENTIAL, intra/inter-op
// 1, ORT_ENABLE_ALL, opset 17) over the small inputs built below; the
// generator is the ignored .agent/voice-probe/gen_lstm_small.py. Sequence
// lengths are int32 because ORT refuses int64 there.
public class CpuExecutionProviderLstmTests
{
    const double Tol = 1e-5;

    static DenseTensor<float> FT(float[] values, params int[] dims) =>
        new DenseTensor<float>(values, dims);

    static float[] Xv() => new float[] { 0.5f, -0.5f, 0.25f, 0.75f };
    static int[] Xd() => new int[] { 2, 1, 2 };
    static float[] Wv() => new float[]
    {
        0.1f, 0.2f, -0.3f, 0.4f, 0.5f, -0.6f, -0.7f, 0.8f,
        0.15f, -0.25f, 0.35f, -0.45f, -0.55f, 0.65f, 0.75f, -0.85f,
    };
    static float[] Rv() => new float[]
    {
        0.2f, -0.1f, 0.3f, 0.5f, -0.4f, 0.6f, 0.7f, -0.8f,
        -0.15f, 0.25f, -0.35f, 0.45f, 0.55f, -0.65f, -0.75f, 0.85f,
    };
    static float[] Bv() => new float[]
    {
        0.1f, -0.1f, 0.2f, -0.2f, 0.05f, -0.05f, 0.15f, -0.15f,
        0.01f, -0.01f, 0.02f, -0.02f, 0.03f, -0.03f, 0.04f, -0.04f,
    };
    static float[] H0v() => new float[] { 0.1f, -0.2f };
    static float[] C0v() => new float[] { 0.3f, 0.1f };
    static float[] Pv() => new float[] { 0.1f, -0.1f, 0.2f, -0.2f, 0.05f, -0.05f };

    static float[] Halved(float[] v) => v.Select(x => 0.5f * x).ToArray();

    sealed class LstmOpts
    {
        public float[]? B;
        public int[]? Lens;
        public float[]? H0;
        public float[]? C0;
        public float[]? P;
        public string? Direction;
        public string[]? Activations;
        public float[]? Alpha;
        public float? Clip;
        public bool InputForget;
        public int OutputCount = 3;
    }

    static OpResult Run(float[] x, int[] xd, float[] w, int[] wd, float[] r, int[] rd, LstmOpts o)
    {
        var X = FT(x, xd);
        var W = FT(w, wd);
        var R = FT(r, rd);
        ITensor? B = o.B is null ? null : FT(o.B, new[] { wd[0], 8 * 2 });
        ITensor? S = o.Lens is null ? null : new DenseTensor<int>(o.Lens, new[] { o.Lens.Length });
        ITensor? H0 = o.H0 is null ? null : FT(o.H0, new[] { wd[0], xd[1], 2 });
        ITensor? C0 = o.C0 is null ? null : FT(o.C0, new[] { wd[0], xd[1], 2 });
        ITensor? P = o.P is null ? null : FT(o.P, new[] { wd[0], 3 * 2 });
        return CPU.Lstm(X, W, R, B, S, H0, C0, P, o.Direction, o.Activations, o.Alpha, null,
            o.Clip, 2, o.InputForget, 0, o.OutputCount, null, null);
    }

    static OpResult Run(float[] x, int[] xd, float[] w, int[] wd, float[] r, int[] rd, float[] b)
    {
        return Run(x, xd, w, wd, r, rd, new LstmOpts { B = b });
    }

    static float[][] Outputs(OpResult r, int count)
    {
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(count, r.Outputs!.Length);
        var outs = new float[count][];
        for (int i = 0; i < count; i++) outs[i] = ((Tensor<float>)r.Outputs[i]).ToArray();
        return outs;
    }

    static void AssertNear(float[] actual, float[] expected, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) < Tol, what + "[" + i + "] drifted: " + actual[i] + " vs " + expected[i] + ".");
    }

    [Fact]
    public void ForwardBasic_MatchesOrt()
    {
        var o = Outputs(Run(Xv(), Xd(), Wv(), new[] { 1, 8, 2 }, Rv(), new[] { 1, 8, 2 }, Bv()), 3);
        AssertNear(o[0], new float[] { -0.13495068f, 0.05704088f, 0.06371567f, -0.07539268f }, "Y");
        AssertNear(o[1], new float[] { 0.06371567f, -0.07539268f }, "Yh");
        AssertNear(o[2], new float[] { 0.12922503f, -0.14709954f }, "Yc");
    }

    [Fact]
    public void InitialState_MatchesOrt()
    {
        var o = Outputs(Run(Xv(), Xd(), Wv(), new[] { 1, 8, 2 }, Rv(), new[] { 1, 8, 2 }, new LstmOpts { B = Bv(), H0 = H0v(), C0 = C0v() }), 3);
        AssertNear(o[0], new float[] { 0.03230323f, 0.05889474f, 0.13819446f, -0.11631908f }, "Y");
        AssertNear(o[1], new float[] { 0.13819446f, -0.11631908f }, "Yh");
        AssertNear(o[2], new float[] { 0.29669607f, -0.21676102f }, "Yc");
    }

    [Fact]
    public void SequenceLens_FreezesStatesAndZeroesPaddedFrames()
    {
        var o = Outputs(Run(Xv(), Xd(), Wv(), new[] { 1, 8, 2 }, Rv(), new[] { 1, 8, 2 }, new LstmOpts { B = Bv(), Lens = new[] { 1 }, H0 = H0v(), C0 = C0v() }), 3);
        AssertNear(o[0], new float[] { 0.03230323f, 0.05889474f, 0f, 0f }, "Y");
        AssertNear(o[1], new float[] { 0.03230323f, 0.05889474f }, "Yh");
        AssertNear(o[2], new float[] { 0.04989660f, 0.18439621f }, "Yc");
    }

    [Fact]
    public void Peephole_MatchesOrt()
    {
        var o = Outputs(Run(Xv(), Xd(), Wv(), new[] { 1, 8, 2 }, Rv(), new[] { 1, 8, 2 }, new LstmOpts { B = Bv(), H0 = H0v(), C0 = C0v(), P = Pv() }), 3);
        AssertNear(o[0], new float[] { 0.03206320f, 0.05715560f, 0.14277087f, -0.11796851f }, "Y");
        AssertNear(o[1], new float[] { 0.14277087f, -0.11796851f }, "Yh");
        AssertNear(o[2], new float[] { 0.29749081f, -0.21545935f }, "Yc");
    }

    [Fact]
    public void Reverse_MatchesOrt()
    {
        var o = Outputs(Run(Xv(), Xd(), Wv(), new[] { 1, 8, 2 }, Rv(), new[] { 1, 8, 2 }, new LstmOpts { B = Bv(), H0 = H0v(), C0 = C0v(), Direction = "reverse" }), 3);
        AssertNear(o[0], new float[] { 0.11566413f, -0.02198114f, 0.20056140f, -0.18987322f }, "Y");
        AssertNear(o[1], new float[] { 0.11566413f, -0.02198114f }, "Yh");
        AssertNear(o[2], new float[] { 0.18269953f, -0.06536282f }, "Yc");
    }

    [Fact]
    public void Bidirectional_MatchesOrt()
    {
        var w = Wv().Concat(Halved(Wv())).ToArray();
        var r = Rv().Concat(Halved(Rv())).ToArray();
        var b = Bv().Concat(Halved(Bv())).ToArray();
        var h0 = H0v().Concat(Halved(H0v())).ToArray();
        var c0 = C0v().Concat(Halved(C0v())).ToArray();
        var o = Outputs(Run(Xv(), Xd(), w, new[] { 2, 8, 2 }, r, new[] { 2, 8, 2 }, new LstmOpts { B = b, H0 = h0, C0 = c0, Direction = "bidirectional" }), 3);
        AssertNear(o[0], new float[] { 0.03230323f, 0.05889474f, 0.03025197f, 0.00469680f, 0.13819446f, -0.11631908f, 0.11130657f, -0.08722385f }, "Y");
        AssertNear(o[1], new float[] { 0.13819446f, -0.11631908f, 0.03025197f, 0.00469680f }, "Yh");
        AssertNear(o[2], new float[] { 0.29669607f, -0.21676102f, 0.05190494f, 0.01178272f }, "Yc");
    }

    [Fact]
    public void ReluActivations_MatchOrt()
    {
        var o = Outputs(Run(Xv(), Xd(), Wv(), new[] { 1, 8, 2 }, Rv(), new[] { 1, 8, 2 }, new LstmOpts { B = Bv(), Activations = new[] { "Relu", "Tanh", "Tanh" } }), 3);
        AssertNear(o[0], new float[] { -0.01794419f, 0f, 0f, -0.01170706f }, "Y");
        AssertNear(o[1], new float[] { 0f, -0.01170706f }, "Yh");
        AssertNear(o[2], new float[] { 0.13661975f, -0.06091034f }, "Yc");
    }

    [Fact]
    public void LeakyReluAlpha_MatchesOrt()
    {
        var o = Outputs(Run(Xv(), Xd(), Wv(), new[] { 1, 8, 2 }, Rv(), new[] { 1, 8, 2 }, new LstmOpts { B = Bv(), Activations = new[] { "LeakyRelu", "Tanh", "Tanh" }, Alpha = new[] { 0.2f, 0f, 0f } }), 3);
        AssertNear(o[0], new float[] { -0.01794419f, 0.00970348f, -0.00247074f, -0.01102191f }, "Y");
        AssertNear(o[1], new float[] { -0.00247074f, -0.01102191f }, "Yh");
        AssertNear(o[2], new float[] { 0.13509507f, -0.05975334f }, "Yc");
    }

    [Fact]
    public void Clip_MatchesOrt()
    {
        var o = Outputs(Run(Xv(), Xd(), Wv(), new[] { 1, 8, 2 }, Rv(), new[] { 1, 8, 2 }, new LstmOpts { B = Bv(), H0 = H0v(), C0 = C0v(), Clip = 0.25f }), 3);
        AssertNear(o[0], new float[] { 0.02802743f, 0.07035208f, 0.07729613f, -0.03311840f }, "Y");
        AssertNear(o[1], new float[] { 0.07729613f, -0.03311840f }, "Yh");
        AssertNear(o[2], new float[] { 0.16192976f, -0.06118195f }, "Yc");
    }

    [Fact]
    public void InputForget_MatchesOrt()
    {
        var o = Outputs(Run(Xv(), Xd(), Wv(), new[] { 1, 8, 2 }, Rv(), new[] { 1, 8, 2 }, new LstmOpts { B = Bv(), H0 = H0v(), C0 = C0v(), InputForget = true }), 3);
        AssertNear(o[0], new float[] { 0.01706260f, 0.06141679f, 0.13089505f, -0.10693064f }, "Y");
        AssertNear(o[1], new float[] { 0.13089505f, -0.10693064f }, "Yh");
        AssertNear(o[2], new float[] { 0.27900207f, -0.19996536f }, "Yc");
    }

    [Fact]
    public void NoBias_MatchesOrt()
    {
        var o = Outputs(Run(Xv(), Xd(), Wv(), new[] { 1, 8, 2 }, Rv(), new[] { 1, 8, 2 }, new LstmOpts()), 3);
        AssertNear(o[0], new float[] { -0.16233273f, 0.08591857f, -0.00730866f, -0.00665735f }, "Y");
        AssertNear(o[1], new float[] { -0.00730866f, -0.00665735f }, "Yh");
        AssertNear(o[2], new float[] { -0.01631326f, -0.01188101f }, "Yc");
    }

    [Fact]
    public void SingleOutput_ReturnsYOnly()
    {
        var o = Outputs(Run(Xv(), Xd(), Wv(), new[] { 1, 8, 2 }, Rv(), new[] { 1, 8, 2 }, new LstmOpts { B = Bv(), OutputCount = 1 }), 1);
        AssertNear(o[0], new float[] { -0.13495068f, 0.05704088f, 0.06371567f, -0.07539268f }, "Y");
    }

    [Fact]
    public void UnknownDirection_Fails()
    {
        var r = Run(Xv(), Xd(), Wv(), new[] { 1, 8, 2 }, Rv(), new[] { 1, 8, 2 }, new LstmOpts { B = Bv(), Direction = "sideways" });
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void WrongActivationCount_Fails()
    {
        var r = Run(Xv(), Xd(), Wv(), new[] { 1, 8, 2 }, Rv(), new[] { 1, 8, 2 }, new LstmOpts { B = Bv(), Activations = new[] { "Sigmoid", "Tanh" } });
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void UnknownActivation_Fails()
    {
        var r = Run(Xv(), Xd(), Wv(), new[] { 1, 8, 2 }, Rv(), new[] { 1, 8, 2 }, new LstmOpts { B = Bv(), Activations = new[] { "Swish", "Tanh", "Tanh" } });
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void BatchFirstLayout_FailsLikeOrt()
    {
        var X = FT(Xv(), new[] { 1, 2, 2 });
        var r = CPU.Lstm(X, FT(Wv(), new[] { 1, 8, 2 }), FT(Rv(), new[] { 1, 8, 2 }), null, null, null, null, null,
            null, null, null, null, null, 2, false, 1, 3, null, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void NonFloatInputs_Fail()
    {
        var X = new DenseTensor<double>(new double[] { 0.5, -0.5, 0.25, 0.75 }, new[] { 2, 1, 2 });
        var r = CPU.Lstm(X, FT(Wv(), new[] { 1, 8, 2 }), FT(Rv(), new[] { 1, 8, 2 }), null, null, null, null, null,
            null, null, null, null, null, 2, false, 0, 3, null, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void LensLengthMismatch_Fails()
    {
        var r = Run(Xv(), Xd(), Wv(), new[] { 1, 8, 2 }, Rv(), new[] { 1, 8, 2 }, new LstmOpts { B = Bv(), Lens = new[] { 1, 2 } });
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void NegativeLens_Fails()
    {
        var r = Run(Xv(), Xd(), Wv(), new[] { 1, 8, 2 }, Rv(), new[] { 1, 8, 2 }, new LstmOpts { B = Bv(), Lens = new[] { -1 } });
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void HiddenSizeMismatch_Fails()
    {
        var r = CPU.Lstm(FT(Xv(), Xd()), FT(Wv(), new[] { 1, 8, 2 }), FT(Rv(), new[] { 1, 8, 2 }), null, null, null, null, null,
            null, null, null, null, null, 3, false, 0, 3, null, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void ReverseShortLength_MatchesOrt()
    {
        var X = FT(new float[] { 0.5f, 9f }, new[] { 2, 1, 1 });
        var W = FT(new float[] { 1f, 1f, 1f, 1f }, new[] { 1, 4, 1 });
        var R = FT(new float[] { 0f, 0f, 0f, 0f }, new[] { 1, 4, 1 });
        var S = new DenseTensor<int>(new[] { 1 }, new[] { 1 });
        var r = CPU.Lstm(X, W, R, null, S, null, null, null, "reverse", null, null, null, null, 1, false, 0, 1, null, null);
        var o = Outputs(r, 1);
        AssertNear(o[0], new float[] { 0.1742697f, 0f }, "Y");
    }

    [Fact]
    public void HardSigmoidDefaults_MatchOrt()
    {
        var X = FT(new float[] { 0.5f, 9f }, new[] { 2, 1, 1 });
        var W = FT(new float[] { 1f, 1f, 1f, 1f }, new[] { 1, 4, 1 });
        var R = FT(new float[] { 0f, 0f, 0f, 0f }, new[] { 1, 4, 1 });
        var S = new DenseTensor<int>(new[] { 2 }, new[] { 1 });
        var r = CPU.Lstm(X, W, R, null, S, null, null, null, "forward", new[] { "HardSigmoid", "Tanh", "Tanh" }, null, null, null, 1, false, 0, 1, null, null);
        var o = Outputs(r, 1);
        AssertNear(o[0], new float[] { 0.16222608f, 0.8557559f }, "Y");
    }

    [Fact]
    public void OverlongLens_FailsLikeOrt()
    {
        var r = Run(Xv(), Xd(), Wv(), new[] { 1, 8, 2 }, Rv(), new[] { 1, 8, 2 }, new LstmOpts { B = Bv(), Lens = new[] { 3 } });
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void ZeroLens_ZeroesOutputsAndStates()
    {
        var o = Outputs(Run(Xv(), Xd(), Wv(), new[] { 1, 8, 2 }, Rv(), new[] { 1, 8, 2 }, new LstmOpts { B = Bv(), Lens = new[] { 0 }, H0 = H0v(), C0 = C0v() }), 3);
        AssertNear(o[0], new float[] { 0f, 0f, 0f, 0f }, "Y");
        AssertNear(o[1], new float[] { 0f, 0f }, "Yh");
        AssertNear(o[2], new float[] { 0f, 0f }, "Yc");
    }

}
