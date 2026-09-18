using System;
using System.Collections.Generic;
using System.Linq;
using Lokad.Onnx;

namespace Lokad.Onnx.Bench;

public sealed record ConvLayer(string Name, int N, int C, int H, int W, int M, int KH, int KW, int SH, int SW, int Pad, bool HasBias, bool FuseRelu, bool ExpectBlocked, string Source);

public sealed record ConvLayerResult(string Name, bool Admitted, string LegacyRoute, string BlockedRoute, double MaxScaled, bool Pass);

// W4 layer comparison vehicle: legacy provider dispatch vs blocked prototype,
// agreement plus routes, no timing. Blocked admission is narrow by design (v1);
// declined layers must still run the real legacy path (route recorded, no agreement).
public static class ConvLayerMatrix
{
    public const double Tolerance = 1e-4;

    static DenseTensor<float> Fill(int[] dims, int off)
    {
        int n = 1;
        foreach (var d in dims) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(((i + off) % 97) - 48) * 0.01f;
        return new DenseTensor<float>(data, dims);
    }

    static (OpResult Result, System.Collections.Generic.IReadOnlyDictionary<string, long> Routes) Profiled(System.Func<OpResult> op)
    {
        using var profilerScope = Profiler.BeginExecution(true);
        Profiler.StartNodeProfile(1, OpType.Conv);
        var r = op();
        Profiler.StopNodeProfile();
        return (r, Profiler.RouteCountsSnapshot());
    }

    static string SoloRoute(System.Collections.Generic.IReadOnlyDictionary<string, long> snap)
    {
        if (snap.Count == 0) return "none";
        return string.Join("+", snap.Keys.OrderBy(k => k, StringComparer.Ordinal));
    }

    public static List<ConvLayerResult> Run(IEnumerable<ConvLayer> layers)
    {
        var results = new List<ConvLayerResult>();
        foreach (var l in layers) results.Add(RunOne(l));
        return results;
    }

    static ConvLayerResult RunOne(ConvLayer l)
    {
        var x = Fill(new int[] { l.N, l.C, l.H, l.W }, 3);
        var w = Fill(new int[] { l.M, l.C, l.KH, l.KW }, 11);
        Tensor<float>? b = l.HasBias ? Fill(new int[] { l.M }, 5) : null;
        int[] pads = l.Pad == 0 ? new int[] { 0, 0, 0, 0 } : new int[] { l.Pad, l.Pad, l.Pad, l.Pad };
        int[] ks = new int[] { l.KH, l.KW };
        int[] st = new int[] { l.SH, l.SW };
        var legacyRun = Profiled(() => CPUExecutionProvider.Conv(x, w, b, "NOTSET", null, 1, ks, pads, st, ExecutionOptions.Default, l.FuseRelu, null));
        if (legacyRun.Result.Status != OpStatus.Success) throw new InvalidOperationException(l.Name + ": legacy conv failed");
        var legacySnap = legacyRun.Routes;
        string legacyRoute = SoloRoute(legacySnap);
        Tensor<float>? blocked = null;
        var blockedRun = Profiled(() =>
        {
            bool ok = Tensor<float>.TryConvBlocked2D(x, w, b, 1, pads, null, st, null, TensorExecutionOptions.Auto, l.FuseRelu, null, out blocked);
            if (!ok) blocked = null;
            return OpResult.Success(OpType.Conv, x);
        });
        var blockedSnap = blockedRun.Routes;
        string blockedRoute = SoloRoute(blockedSnap);
        bool pass;
        double worst = 0.0;
        if (!l.ExpectBlocked)
        {
            pass = blocked is null && !blockedSnap.ContainsKey("conv-blocked") && legacyRoute != "none";
        }
        else
        {
            pass = false;
            if (blocked is not null && blockedRoute.Contains("conv-blocked"))
            {
                var ya = blocked.ToArray();
                var xa = x.ToArray();
                var wa = w.ToArray();
                var ba = b?.ToArray();
                int outH = (l.H + 2 * l.Pad - l.KH) / l.SH + 1;
                int outW = (l.W + 2 * l.Pad - l.KW) / l.SW + 1;
                worst = 0.0;
                bool shapeOk = ya.Length == l.M * outH * outW;
                for (int m = 0; m < l.M && shapeOk; m++)
                    for (int oh = 0; oh < outH; oh++)
                        for (int ow = 0; ow < outW; ow++)
                        {
                            double acc = ba is null ? 0.0 : ba[m];
                            for (int c = 0; c < l.C; c++)
                                for (int kh = 0; kh < l.KH; kh++)
                                {
                                    int ih = oh * l.SH + kh - l.Pad;
                                    if (ih < 0 || ih >= l.H) continue;
                                    for (int kw = 0; kw < l.KW; kw++)
                                    {
                                        int iw = ow * l.SW + kw - l.Pad;
                                        if (iw < 0 || iw >= l.W) continue;
                                        acc += xa[(c * l.H + ih) * l.W + iw] * wa[((m * l.C + c) * l.KH + kh) * l.KW + kw];
                                    }
                                }
                            if (l.FuseRelu && acc < 0.0) acc = 0.0;
                            double got = ya[(m * outH + oh) * outW + ow];
                            double d = Math.Abs(got - acc) / Math.Max(1.0, Math.Abs(acc));
                            if (d > worst) worst = d;
                        }
                pass = shapeOk && worst <= Tolerance;
            }
        }
        return new ConvLayerResult(l.Name, blocked is not null, legacyRoute, blockedRoute, worst, pass);
    }
}
