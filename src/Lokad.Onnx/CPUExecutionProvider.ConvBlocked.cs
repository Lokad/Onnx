namespace Lokad.Onnx;

using System;
using System.Buffers;
using System.Collections.Generic;

using static OpResult;

public partial class CPUExecutionProvider
{
    /// <summary>Executes one blocked residual region: convert the input once, run every step on blocked buffers, convert the output once.</summary>
    /// <remarks>Geometry comes from the live tensors every run (nothing baked except the step plan), so varying batch and spatial sizes work. Filters resolve from prepared packs with per-run source validation and repack on mismatch; biases and skips read current values every run. Intermediate blocked buffers rent from the shared array pool and return before exit.</remarks>
    internal static OpResult RunBlockedRegion(BlockedRegionSpec spec, ITensor? input, ComputationalGraph graph, ExecutionOptions? options)
    {
        var op = OpType.Conv;
        if (input is not Tensor<float> xd) return input is null ? MissingInput(op, nameof(input)) : WrongInputType(op, nameof(input), TensorElementType.Float, input);
        if (xd.Rank != 4) return WrongInputShape(op, nameof(input), 4, input);
        var opts = (options ?? ExecutionOptions.Default).Validated();
        var ddims = xd.Dimensions;
        int n = ddims[0], c0 = ddims[1], h = ddims[2], w = ddims[3];
        if (n < 1 || c0 < 16 || c0 % 16 != 0 || h < 1 || w < 1) return WrongInputShape(op, nameof(input), input, "Blocked regions need a positive batch and spatial extent with 16-multiple channels.");
        DenseTensor<float> xdense = xd as DenseTensor<float> ?? xd.ToDenseTensor();
        var rented = new List<float[]>();
        try
        {
            float[] yFlat = Array.Empty<float>();
            int yChannels = 0;
            int sampleIn = c0 * h * w;
            for (int s = 0; s < n; s++)
            {
                float[] sample = RunBlockedSample(spec, graph, xdense.Buffer.Span.Slice(s * sampleIn, sampleIn), c0, h, w, opts, s, n, rented);
                if (s == 0)
                {
                    if (sample.Length % (h * w) != 0) return Failure(op, "Blocked region produced a misaligned sample.");
                    yChannels = sample.Length / (h * w);
                    yFlat = new float[n * sample.Length];
                }
                if (sample.Length != yChannels * h * w) return Failure(op, "Blocked region produced an inconsistent sample shape.");
                Array.Copy(sample, 0, yFlat, s * sample.Length, sample.Length);
            }
            if (yChannels == 0) return Failure(op, "Blocked region produced no output.");
            return Success(op, new DenseTensor<float>(new Memory<float>(yFlat), new[] { n, yChannels, h, w }));
        }
        finally
        {
            foreach (var buf in rented) ArrayPool<float>.Shared.Return(buf);
        }
    }

    static float[] RunBlockedSample(
        BlockedRegionSpec spec,
        ComputationalGraph graph,
        ReadOnlySpan<float> xs,
        int c,
        int h,
        int w,
        ExecutionOptions opts,
        int sample,
        int batch,
        List<float[]> rented)
    {
        var blocked = new Dictionary<string, float[]>(StringComparer.Ordinal);
        float[] BindBlocked(string name, int channels)
        {
            var buf = ArrayPool<float>.Shared.Rent(channels / 16 * h * w * 16);
            rented.Add(buf);
            blocked[name] = buf;
            return buf;
        }
        float[] ResolveBlocked(string name, int channels, int s, int n)
        {
            if (blocked.TryGetValue(name, out var hit)) return hit;
            if (!TryResolveValue(graph, name, out var t) || t is not Tensor<float> tf) throw new InvalidOperationException("Blocked region input " + name + " did not resolve to a float tensor.");
            var dd = tf as DenseTensor<float> ?? tf.ToDenseTensor();
            var ddims = dd.Dimensions;
            if (dd.Rank != 4 || ddims[1] != channels || ddims[2] != h || ddims[3] != w) throw new InvalidOperationException("Blocked region input " + name + " has an incompatible shape.");
            int per = channels * h * w;
            int at = (ddims[0] > 1 ? s : 0) * per;
            if (ddims[0] != 1 && ddims[0] != n) throw new InvalidOperationException("Blocked region input " + name + " has an incompatible batch.");
            float[] buf = BindBlocked(name, channels);
            MathOpsConvBlocked.BlockInput(dd.Buffer.Span.Slice(at, per), buf.AsSpan(0, per), channels, h, w);
            return buf;
        }
        int curChannels = c;
        {
            float[] xb = BindBlocked(spec.InputName, c);
            MathOpsConvBlocked.BlockInput(xs, xb.AsSpan(0, c / 16 * h * w * 16), c, h, w);
        }
        foreach (var step in spec.Steps)
        {
            if (step.Kind == BlockedRegionStepKind.Conv && step.FilterName is not null)
            {
                float[] filt = ResolveFilter(graph, step.FilterName, out int m, out int wc);
                if (wc != curChannels) throw new InvalidOperationException("Blocked region channel mismatch at " + step.Output + ".");
                float[] outB = BindBlocked(step.Output, m);
                if (!blocked.TryGetValue(step.DataInput, out var cdin)) throw new InvalidOperationException("Blocked region value " + step.DataInput + " was not produced.");
                MathOpsConvBlocked.RunBlockedConv(cdin, filt, outB, curChannels / 16, m / 16, h, w, opts.Tensor);
                if (step.BiasName is not null)
                {
                    if (!TryResolveValue(graph, step.BiasName, out var bt) || bt is not Tensor<float> btf) throw new InvalidOperationException("Blocked region bias " + step.BiasName + " did not resolve.");
                    var bd = btf as DenseTensor<float> ?? btf.ToDenseTensor();
                    if (bd.Length != m) throw new InvalidOperationException("Blocked region bias " + step.BiasName + " has an incompatible shape.");
                    MathOpsConvBlocked.BlockedBiasAdd(outB.AsSpan(0, m * h * w), bd.Buffer.Span, outB.AsSpan(0, m * h * w), h * w, opts.Tensor);
                }
                if (step.ReluAfter) MathOpsConvBlocked.BlockedRelu(outB.AsSpan(0, m * h * w), opts.Tensor);
                curChannels = m;
            }
            else if (step.Kind == BlockedRegionStepKind.Add && step.AuxInput is not null)
            {
                if (!blocked.TryGetValue(step.DataInput, out var a)) throw new InvalidOperationException("Blocked region value " + step.DataInput + " was not produced.");
                float[] b = ResolveBlocked(step.AuxInput, curChannels, sample, batch);
                float[] outB = BindBlocked(step.Output, curChannels);
                int addLen = curChannels * h * w;
                MathOpsConvBlocked.BlockedAdd(a.AsSpan(0, addLen), b.AsSpan(0, addLen), outB.AsSpan(0, addLen), opts.Tensor);
            }
            else if (step.Kind == BlockedRegionStepKind.Relu)
            {
                if (!blocked.TryGetValue(step.DataInput, out var d)) throw new InvalidOperationException("Blocked region value " + step.DataInput + " was not produced.");
                MathOpsConvBlocked.BlockedRelu(d, opts.Tensor);
                blocked[step.Output] = d;
            }
            else throw new InvalidOperationException("Blocked region step is malformed.");
        }
        if (!blocked.TryGetValue(spec.OutputName, out var fin)) throw new InvalidOperationException("Blocked region output was not produced.");
        var ys = new float[curChannels * h * w];
        MathOpsConvBlocked.UnblockOutput(fin, ys, null, curChannels, h, w, false);
        return ys;
    }

    static bool TryResolveValue(ComputationalGraph graph, string name, out ITensor? value)
    {
        value = null;
        try
        {
            value = graph.GetInputTensor(name);
            return value is not null;
        }
        catch (Exception)
        {
            return false;
        }
    }

    static float[] ResolveFilter(ComputationalGraph graph, string name, out int m, out int c)
    {
        if (graph.BlockedFilters.TryGetValue(name, out var prep)
            && graph.Initializers.TryGetValue(name, out var cur)
            && ReferenceEquals(cur, prep.SourceRef) && cur.Length == prep.SourceLength
            && cur is Tensor<float> ctf && ctf.Rank == 4 && ctf.Dimensions.Length == 4)
        {
            m = prep.M;
            c = prep.C;
            return prep.Packed;
        }
        if (!graph.Initializers.TryGetValue(name, out var src) || src is not Tensor<float> stf || stf.Rank != 4)
            throw new InvalidOperationException("Blocked region filter " + name + " did not resolve.");
        var sd = stf.Dimensions;
        m = sd[0];
        c = sd[1];
        if (sd.Length != 4 || sd[2] != 3 || sd[3] != 3 || m % 16 != 0 || c % 16 != 0)
            throw new InvalidOperationException("Blocked region filter " + name + " has an incompatible shape.");
        float[] packed = src is DenseTensor<float> sdd
            ? MathOpsConvBlocked.PackBlockedFilter(sdd.Buffer.Span, m, c)
            : MathOpsConvBlocked.PackBlockedFilter(stf.ToArray(), m, c);
        graph.BlockedFilters[name] = new PreparedBlockedFilter(packed, src, src.Length, m, c);
        return packed;
    }
}

