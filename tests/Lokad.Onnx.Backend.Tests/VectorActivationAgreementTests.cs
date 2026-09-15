using System;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins the vector-activation paths against the scalar nests they replace:
/// the elementwise Sigmoid/Tanh providers and the LSTM default-trio fast gate
/// must agree with the scalar contracts within float rounding. Scalar
/// ExecutionOptions force the legacy nests; Default enables the spans.
/// </summary>
public class VectorActivationAgreementTests
{
    static void AssertScaledNear(float[] expected, float[] actual, double tol, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        int at = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            if (float.IsNaN(e) && float.IsNaN(a)) continue;
            Assert.True(!float.IsNaN(e) && !float.IsNaN(a), what + " NaN mismatch at " + i);
            if (float.IsInfinity(e) || float.IsInfinity(a))
            {
                Assert.True(e == a, what + " infinite mismatch at " + i + ": " + a + " vs " + e);
                continue;
            }
            double err = Math.Abs((double)a - e) / (1.0 + Math.Abs((double)e));
            if (err > worst) { worst = err; at = i; }
        }
        Assert.True(worst <= tol, what + " worst=" + worst.ToString("E2") + " at " + at);
    }

    static float[] RunOne(Func<ExecutionOptions?, OpResult> run, ExecutionOptions? opt)
    {
        var r = run(opt);
        Assert.Equal(OpStatus.Success, r.Status);
        return ((Tensor<float>)r.Outputs[0]).ToArray();
    }

    [Fact]
    public void SigmoidProvider_DefaultMatchesScalar()
    {
        var rnd = new Random(41);
        int n = 1000;
        var vals = new float[n];
        for (int i = 0; i < n; i++) vals[i] = (float)(rnd.NextDouble() * 40 - 20);
        var x = new DenseTensor<float>(vals, new[] { n });
        var expected = RunOne(o => CPUExecutionProvider.Sigmoid(x, o), ExecutionOptions.Scalar);
        var actual = RunOne(o => CPUExecutionProvider.Sigmoid(x, o), null);
        AssertScaledNear(expected, actual, 1e-6, "sigmoid-provider");
    }

    [Fact]
    public void TanhProvider_DefaultMatchesScalar()
    {
        var rnd = new Random(43);
        int n = 1000;
        var vals = new float[n];
        for (int i = 0; i < n; i++) vals[i] = (float)(rnd.NextDouble() * 40 - 20);
        var x = new DenseTensor<float>(vals, new[] { n });
        var expected = RunOne(o => CPUExecutionProvider.Tanh(x, o), ExecutionOptions.Scalar);
        var actual = RunOne(o => CPUExecutionProvider.Tanh(x, o), null);
        AssertScaledNear(expected, actual, 1e-6, "tanh-provider");
    }

    [Fact]
    public void LstmDefaultTrio_DefaultMatchesScalar()
    {
        // H=32 exercises the vector lanes plus tail width; seq=5 with two
        // batches keeps recurrent drift bounded while covering the gate math.
        int seq = 5, batch = 2, input = 7, h = 32;
        var rnd = new Random(47);
        float Next() => (float)(rnd.NextDouble() * 1.0 - 0.5);
        var xv = new float[seq * batch * input];
        var wv = new float[4 * h * input];
        var rv = new float[4 * h * h];
        var bv = new float[8 * h];
        for (int i = 0; i < xv.Length; i++) xv[i] = Next();
        for (int i = 0; i < wv.Length; i++) wv[i] = Next();
        for (int i = 0; i < rv.Length; i++) rv[i] = Next();
        for (int i = 0; i < bv.Length; i++) bv[i] = Next();
        var X = new DenseTensor<float>(xv, new[] { seq, batch, input });
        var W = new DenseTensor<float>(wv, new[] { 1, 4 * h, input });
        var R = new DenseTensor<float>(rv, new[] { 1, 4 * h, h });
        var B = new DenseTensor<float>(bv, new[] { 1, 8 * h });
        float[][] RunAll(ExecutionOptions? opt)
        {
            var r = CPUExecutionProvider.Lstm(X, W, R, B, null, null, null, null, null, null, null, null, null, h, false, 0, 3, opt, null);
            Assert.Equal(OpStatus.Success, r.Status);
            Assert.Equal(3, r.Outputs.Length);
            return new[]
            {
                ((Tensor<float>)r.Outputs[0]).ToArray(),
                ((Tensor<float>)r.Outputs[1]).ToArray(),
                ((Tensor<float>)r.Outputs[2]).ToArray(),
            };
        }
        var expected = RunAll(ExecutionOptions.Scalar);
        var actual = RunAll(null);
        string[] names = new[] { "lstm-Y", "lstm-Yh", "lstm-Yc" };
        for (int k = 0; k < 3; k++) AssertScaledNear(expected[k], actual[k], 1e-5, names[k]);
    }
}
