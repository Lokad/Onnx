using Xunit;

namespace Lokad.Onnx.Backend.Tests;

// P12: the vectorized float LayerNorm must match an independent double
// reference within float tolerance on odd blocks, near-constant and large
// values, with and without bias, through both entries.
public class LayerNormSimdTests
{
    static float[] IndependentLayerNorm(float[,] x, float[] gamma, float[]? bias, double eps)
    {
        int rows = x.GetLength(0), cols = x.GetLength(1);
        var y = new float[rows * cols];
        for (int i = 0; i < rows; i++)
        {
            double mean = 0;
            for (int j = 0; j < cols; j++) mean += x[i, j];
            mean /= cols;
            double variance = 0;
            for (int j = 0; j < cols; j++) variance += (x[i, j] - mean) * (x[i, j] - mean);
            variance /= cols;
            double denom = System.Math.Sqrt(variance + eps);
            for (int j = 0; j < cols; j++)
            {
                double b = bias is null ? 0.0 : bias[j];
                y[i * cols + j] = (float)((x[i, j] - mean) / denom * gamma[j] + b);
            }
        }
        return y;
    }

    static DenseTensor<float> Dense(float[,] v) => DenseTensor<float>.OfValues(v);

    static void Agrees(float[,] x, float[] gamma, float[]? bias, double eps)
    {
        int rows = x.GetLength(0), cols = x.GetLength(1);
        var g = Dense(new float[1, cols]);
        for (int j = 0; j < cols; j++) g.SetValue(j, gamma[j]);
        DenseTensor<float>? b = null;
        if (bias is not null)
        {
            b = Dense(new float[1, cols]);
            for (int j = 0; j < cols; j++) b.SetValue(j, bias[j]);
        }
        var t = Dense(x);
        var actual = Tensor<float>.LayerNormalization(t, g, b, -1, (float)eps).ToArray();
        var expected = IndependentLayerNorm(x, gamma, bias, eps);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 5);
        var dest = Tensor<float>.Zeros(rows, cols).ToDenseTensor();
        Tensor<float>.LayerNormalization(t, g, b, dest, -1, (float)eps);
        Assert.Equal(actual, dest.ToArray());
    }

    [Fact]
    public void BasicWithBias()
    {
        Agrees(new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } },
            new[] { 1f, 2f, 3f, 4f }, new[] { 0.5f, -0.5f, 1f, 0f }, 1e-5);
    }

    [Fact]
    public void OddBlockNoBias()
    {
        var x = new float[3, 10];
        var g = new float[10];
        var rnd = new System.Random(17);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 10; j++)
                x[i, j] = (float)rnd.NextDouble() * 4f - 2f;
        for (int j = 0; j < 10; j++) g[j] = (float)rnd.NextDouble() + 0.5f;
        Agrees(x, g, null, 1e-5);
    }

    [Fact]
    public void NearConstantValues()
    {
        var x = new float[2, 16];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 16; j++)
                x[i, j] = 1f + (j % 2 == 0 ? 1e-7f : -1e-7f);
        var g = new float[16];
        for (int j = 0; j < 16; j++) g[j] = 1f;
        Agrees(x, g, null, 1e-12);
    }

    [Fact]
    public void LargeValues()
    {
        var x = new float[2, 12];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 12; j++)
                x[i, j] = 1e10f + j * 1e4f;
        var g = new float[12];
        for (int j = 0; j < 12; j++) g[j] = 1f;
        Agrees(x, g, g, 1e-5);
    }
}
