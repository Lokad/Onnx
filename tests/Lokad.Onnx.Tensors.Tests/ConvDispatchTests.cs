using System;
using System.Linq;
using Xunit;

namespace Lokad.Onnx.Tensors.Tests;

public class ConvDispatchTests
{
    static System.Reflection.MethodInfo FindConvOptions(bool dbl, bool pads)
    {
        var type = dbl ? typeof(Tensor<double>) : typeof(Tensor<float>);
        var tensor = dbl ? typeof(Tensor<double>) : typeof(Tensor<float>);
        var hit = type.GetMethods().FirstOrDefault(m =>
            m.Name == "Conv2D" &&
            m.GetParameters().Length > 4 &&
            m.GetParameters()[0].ParameterType == tensor &&
            m.GetParameters().Any(p => p.ParameterType == typeof(TensorExecutionOptions)) &&
            m.GetParameters()[3].ParameterType == (pads ? typeof(int[]) : typeof(MathOps.PadType)));
        Assert.NotNull(hit);
        return hit!;
    }

    static System.Reflection.MethodInfo FindDoubleMatMul2DDestination()
    {
        var hit = typeof(Tensor<double>).GetMethods().FirstOrDefault(m =>
            m.Name == "MatMul2D" &&
            m.GetParameters().Select(p => p.ParameterType).SequenceEqual(
                new[] { typeof(Tensor<double>), typeof(Tensor<double>), typeof(DenseTensor<double>), typeof(TensorExecutionOptions) }));
        Assert.NotNull(hit);
        return hit!;
    }

    static Tensor<float> RunFloatConv(System.Reflection.MethodInfo m, Tensor<float> x, Tensor<float> w, int group, TensorExecutionOptions opts)
    {
        var ps = m.GetParameters();
        var args = new object?[ps.Length];
        args[0] = x; args[1] = w; args[2] = group;
        for (int i = 3; i < ps.Length - 1; i++) args[i] = ps[i].ParameterType.IsValueType && Nullable.GetUnderlyingType(ps[i].ParameterType) is null ? Activator.CreateInstance(ps[i].ParameterType) : null;
        args[ps.Length - 1] = opts;
        if (ps[3].ParameterType == typeof(MathOps.PadType)) args[3] = MathOps.PadType.Valid;
        else args[3] = new int[] { 0, 0, 0, 0 };
        return (Tensor<float>)m.Invoke(null, args)!;
    }


    [Fact]
    public void DispatchEntries_Exist()
    {
        FindDoubleMatMul2DDestination();
        FindConvOptions(false, false);
        FindConvOptions(false, true);
        FindConvOptions(true, false);
        FindConvOptions(true, true);
    }

    [Fact]
    public void Conv_ParallelMatchesSequential_Float()
    {
        var m = FindConvOptions(false, false);
        var rnd = new System.Random(21);
        Tensor<float> Rand(params int[] dims)
        {
            int n = 1;
            foreach (var d in dims) n *= d;
            var a = new float[n];
            for (int i = 0; i < n; i++) a[i] = (float)rnd.NextDouble() - 0.5f;
            return new DenseTensor<float>(a, dims);
        }
        foreach (var (xd, wd, g) in new[] { (new[] { 1, 4, 8, 8 }, new[] { 4, 4, 3, 3 }, 1), (new[] { 1, 8, 8, 8 }, new[] { 8, 2, 3, 3 }, 4), (new[] { 1, 4, 8, 8 }, new[] { 4, 1, 3, 3 }, 4) })
        {
            var x = Rand(xd);
            var w = Rand(wd);
            var seq = RunFloatConv(m, x, w, g, TensorExecutionOptions.Scalar);
            var par = RunFloatConv(m, x, w, g, TensorExecutionOptions.Parallel(2));
            Assert.Equal(seq.Dimensions.ToArray(), par.Dimensions.ToArray());
            var sa = seq.ToArray();
            var pa = par.ToArray();
            for (int i = 0; i < sa.Length; i++) Assert.Equal(sa[i], pa[i], 5);
            var plain = Tensor<float>.Conv2D(x, w, g, MathOps.PadType.Valid, null, null, null, null, null);
            var pla = plain.ToArray();
            for (int i = 0; i < sa.Length; i++) Assert.Equal(sa[i], pla[i], 5);
        }
    }

    [Fact]
    public void Conv_BiasMatchesUnbiasedPlusBias_Float()
    {
        var m = FindConvOptions(false, true);
        var x = Tensor<float>.Arange(0f, 2f * 2f * 4f * 4f).Reshape(2, 2, 4, 4);
        var w = Tensor<float>.Ones(2, 2, 2, 2);
        var bias = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        var pads = new int[] { 0, 0, 0, 0 };
        var withBias = (Tensor<float>)m.Invoke(null, new object?[] { x, w, 1, pads, bias, null, null, null, TensorExecutionOptions.Scalar })!;
        var plain = (Tensor<float>)m.Invoke(null, new object?[] { x, w, 1, pads, null, null, null, null, TensorExecutionOptions.Scalar })!;
        Assert.Equal(plain.Dimensions.ToArray(), withBias.Dimensions.ToArray());
        var pa = plain.ToArray();
        var ba = withBias.ToArray();
        for (int i = 0; i < ba.Length; i++)
        {
            int channel = (i / (3 * 3)) % 2;
            Assert.Equal(pa[i] + (channel == 0 ? 1f : 2f), ba[i], 4);
        }
    }

    [Fact]
    public void DoubleMatMul2D_Destination_WritesProduct()
    {
        var mm = FindDoubleMatMul2DDestination();
        var x = DenseTensor<double>.OfValues(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        var y = DenseTensor<double>.OfValues(new double[,] { { 5.0, 6.0 }, { 7.0, 8.0 } });
        var dest = new DenseTensor<double>(new double[] { 9.0, 9.0, 9.0, 9.0 }, new[] { 2, 2 });
        var r = (Tensor<double>)mm.Invoke(null, new object?[] { x, y, dest, TensorExecutionOptions.Simd })!;
        Assert.Equal(new double[] { 19.0, 22.0, 43.0, 50.0 }, r.ToArray());
    }

    [Fact]
    public void Conv_ParallelMatchesSequential_Double_NoBias()
    {
        var m = FindConvOptions(true, true);
        var x = DenseTensor<double>.OfValues(new double[1, 1, 4, 4]
        { {
            { { 0.0, 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0, 7.0 }, { 8.0, 9.0, 10.0, 11.0 }, { 12.0, 13.0, 14.0, 15.0 } }
        } });
        var w = Tensor<double>.Ones(1, 1, 2, 2);
        var pads = new int[] { 0, 0, 0, 0 };
        var seq = (Tensor<double>)m.Invoke(null, new object?[] { x, w, 1, pads, null, null, null, null, TensorExecutionOptions.Scalar })!;
        var par = (Tensor<double>)m.Invoke(null, new object?[] { x, w, 1, pads, null, null, null, null, TensorExecutionOptions.Parallel(2) })!;
        Assert.Equal(new[] { 1, 1, 3, 3 }, seq.Dimensions.ToArray());
        var sa = seq.ToArray();
        var pa = par.ToArray();
        for (int i = 0; i < sa.Length; i++) Assert.Equal(sa[i], pa[i], 9);
        Assert.Equal(10.0, seq[0, 0, 0, 0], 9);
        var plain = Tensor<double>.Conv2D(x, w, 1, pads, null, null, null, null);
        var pla = plain.ToArray();
        for (int i = 0; i < sa.Length; i++) Assert.Equal(sa[i], pla[i], 9);
    }

    [Fact]
    public void Conv_FailedRun_DoesNotCorruptRetry()
    {
        var x = Tensor<float>.Ones(1, 1, 4, 4);
        var w = Tensor<float>.Ones(1, 1, 2, 2);
        Assert.ThrowsAny<Exception>(() => Tensor<float>.Conv2D(x, w, 2, MathOps.PadType.Valid, null, null, null, null, null));
        var y = Tensor<float>.Conv2D(x, w, 1, MathOps.PadType.Valid, null, null, null, null, null);
        Assert.Equal(new[] { 1, 1, 3, 3 }, y.Dimensions.ToArray());
        Assert.Equal(4f, y[0, 0, 0, 0], 5);
    }
}
