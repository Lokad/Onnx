namespace Lokad.Onnx;

using System;

using static OpResult;

public partial class CPUExecutionProvider
{
    /// <summary>
    /// Long short-term memory recurrence over float32 tensors, forward,
    /// reverse, and bidirectional. Implements the ONNX gate order (input,
    /// output, forget, cell) over W/R rows, split W/R biases, peephole
    /// weights on the previous cell for the input/forget gates and on the
    /// new cell for the output gate, optional clipping of gate
    /// pre-activations, coupled input/forget gates, and sequence-length
    /// freezing (padded steps emit zero frames while states carry over).
    /// Only seq-first layout is accepted: batch-first layout is refused by
    /// the ORT CPU backend as well, so matching that refusal keeps parity.
    /// Only float32 is computed; every pooled destination is fully assigned
    /// (padded Y steps are zeroed, final states land for every direction and
    /// batch), so plain pooled rents are sound and no accumulation ever
    /// reads uninitialized storage.
    /// </summary>
    public static OpResult Lstm(
        ITensor? X, ITensor? W, ITensor? R, ITensor? B,
        ITensor? sequenceLens, ITensor? initialH, ITensor? initialC, ITensor? P,
        string? direction, string[]? activations, float[]? activationAlpha, float[]? activationBeta,
        float? clip, int hiddenSize, bool inputForget, int layout,
        int outputCount, ExecutionOptions? options, TensorBufferPool? pool)
    {
        var op = OpType.LSTM;
        if (X is null) return MissingInput(op, nameof(X));
        if (W is null) return MissingInput(op, nameof(W));
        if (R is null) return MissingInput(op, nameof(R));
        if (hiddenSize <= 0) return AttributeNotSupported(op, "hidden_size", hiddenSize.ToString(), "hidden_size must be positive.");
        if (layout != 0) return AttributeNotSupported(op, "layout", layout.ToString(), "Only seq-first layout (0) is supported; batch-first layout is refused by the ORT CPU backend as well.");
        int numDirections;
        bool reverse;
        switch ((direction ?? "forward").ToLowerInvariant())
        {
            case "forward": numDirections = 1; reverse = false; break;
            case "reverse": numDirections = 1; reverse = true; break;
            case "bidirectional": numDirections = 2; reverse = false; break;
            default: return AttributeNotSupported(op, "direction", direction ?? "", "direction must be forward, reverse, or bidirectional.");
        }
        if (X.ElementType != TensorElementType.Float) return InputTypeNotSupported(op, nameof(X), X, "Only float32 LSTM inputs are supported.");
        if (W.ElementType != TensorElementType.Float) return InputTypeNotSupported(op, nameof(W), W, "Only float32 LSTM inputs are supported.");
        if (R.ElementType != TensorElementType.Float) return InputTypeNotSupported(op, nameof(R), R, "Only float32 LSTM inputs are supported.");
        if (B is not null && B.ElementType != TensorElementType.Float) return InputTypeNotSupported(op, nameof(B), B, "Only float32 LSTM inputs are supported.");
        if (initialH is not null && initialH.ElementType != TensorElementType.Float) return InputTypeNotSupported(op, nameof(initialH), initialH, "Only float32 LSTM inputs are supported.");
        if (initialC is not null && initialC.ElementType != TensorElementType.Float) return InputTypeNotSupported(op, nameof(initialC), initialC, "Only float32 LSTM inputs are supported.");
        if (P is not null && P.ElementType != TensorElementType.Float) return InputTypeNotSupported(op, nameof(P), P, "Only float32 LSTM inputs are supported.");
        if (X.Rank != 3) return WrongInputShape(op, nameof(X), 3, X);
        int seq = X.Dims[0];
        int batch = X.Dims[1];
        int inputSize = X.Dims[2];
        if (W.Rank != 3 || W.Dims[0] != numDirections || W.Dims[1] != 4 * hiddenSize || W.Dims[2] != inputSize)
            return WrongInputShape(op, nameof(W), W, "W must be [num_directions, 4*hidden_size, input_size].");
        if (R.Rank != 3 || R.Dims[0] != numDirections || R.Dims[1] != 4 * hiddenSize || R.Dims[2] != hiddenSize)
            return WrongInputShape(op, nameof(R), R, "R must be [num_directions, 4*hidden_size, hidden_size].");
        if (B is not null)
        {
            bool okB = (B.Rank == 2 && B.Dims[0] == numDirections && B.Dims[1] == 8 * hiddenSize)
                || (B.Rank == 1 && numDirections == 1 && B.Dims[0] == 8 * hiddenSize);
            if (!okB) return WrongInputShape(op, nameof(B), B, "B must be [num_directions, 8*hidden_size].");
        }
        int[]? lens = null;
        if (sequenceLens is not null)
        {
            if (sequenceLens.Rank != 1 || sequenceLens.Dims[0] != batch)
                return WrongInputShape(op, nameof(sequenceLens), sequenceLens, "sequence_lens must be [batch_size].");
            if (sequenceLens.ElementType != TensorElementType.Int32 && sequenceLens.ElementType != TensorElementType.Int64)
                return WrongInputType(op, nameof(sequenceLens), "sequence_lens must be int32 or int64.", sequenceLens);
            lens = ToIntArray(sequenceLens, nameof(sequenceLens));
            for (int i = 0; i < lens.Length; i++)
                if (lens[i] < 0 || lens[i] > seq) return Failure(op, "sequence_lens values must be within [0, seq_length].");
        }
        if (initialH is not null && (initialH.Rank != 3 || initialH.Dims[0] != numDirections || initialH.Dims[1] != batch || initialH.Dims[2] != hiddenSize))
            return WrongInputShape(op, nameof(initialH), initialH, "initial_h must be [num_directions, batch_size, hidden_size].");
        if (initialC is not null && (initialC.Rank != 3 || initialC.Dims[0] != numDirections || initialC.Dims[1] != batch || initialC.Dims[2] != hiddenSize))
            return WrongInputShape(op, nameof(initialC), initialC, "initial_c must be [num_directions, batch_size, hidden_size].");
        if (P is not null && (P.Rank != 2 || P.Dims[0] != numDirections || P.Dims[1] != 3 * hiddenSize))
            return WrongInputShape(op, nameof(P), P, "P must be [num_directions, 3*hidden_size].");
        int actCount = 3 * numDirections;
        string[] acts = activations ?? Array.Empty<string>();
        if (acts.Length == 0)
        {
            acts = new string[actCount];
            for (int d = 0; d < numDirections; d++)
            {
                acts[3 * d] = "Sigmoid";
                acts[3 * d + 1] = "Tanh";
                acts[3 * d + 2] = "Tanh";
            }
        }
        else if (acts.Length != actCount)
        {
            return AttributeNotSupported(op, "activations", string.Join(",", acts), "activations must hold 3 entries per direction.");
        }
        if (activationAlpha is not null && activationAlpha.Length != actCount)
            return AttributeNotSupported(op, "activation_alpha", activationAlpha.Length.ToString(), "activation_alpha must hold 3 entries per direction.");
        if (activationBeta is not null && activationBeta.Length != actCount)
            return AttributeNotSupported(op, "activation_beta", activationBeta.Length.ToString(), "activation_beta must hold 3 entries per direction.");
        if (outputCount < 1 || outputCount > 3)
            return Failure(op, "LSTM supports 1..3 outputs but the node declares " + outputCount + ".");
        var gateF = new Func<float, float>[numDirections];
        var gateG = new Func<float, float>[numDirections];
        var gateH = new Func<float, float>[numDirections];
        for (int d = 0; d < numDirections; d++)
        {
            float a0 = activationAlpha is null ? LstmDefaultAlpha(acts[3 * d]) : activationAlpha[3 * d];
            float b0 = activationBeta is null ? LstmDefaultBeta(acts[3 * d]) : activationBeta[3 * d];
            float a1 = activationAlpha is null ? LstmDefaultAlpha(acts[3 * d + 1]) : activationAlpha[3 * d + 1];
            float b1 = activationBeta is null ? LstmDefaultBeta(acts[3 * d + 1]) : activationBeta[3 * d + 1];
            float a2 = activationAlpha is null ? LstmDefaultAlpha(acts[3 * d + 2]) : activationAlpha[3 * d + 2];
            float b2 = activationBeta is null ? LstmDefaultBeta(acts[3 * d + 2]) : activationBeta[3 * d + 2];
            var f = LstmActivation(acts[3 * d], a0, b0);
            var g = LstmActivation(acts[3 * d + 1], a1, b1);
            var h = LstmActivation(acts[3 * d + 2], a2, b2);
            if (f is null) return AttributeNotSupported(op, "activations", acts[3 * d], "Unknown LSTM gate activation.");
            if (g is null) return AttributeNotSupported(op, "activations", acts[3 * d + 1], "Unknown LSTM gate activation.");
            if (h is null) return AttributeNotSupported(op, "activations", acts[3 * d + 2], "Unknown LSTM gate activation.");
            gateF[d] = f;
            gateG[d] = g;
            gateH[d] = h;
        }
        var opts = (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.Math);
        var xd = Tensor<float>.RequireContiguous((Tensor<float>)X, nameof(X), opts.Tensor.CopyReporter);
        var wd = Tensor<float>.RequireContiguous((Tensor<float>)W, nameof(W), opts.Tensor.CopyReporter);
        var rd = Tensor<float>.RequireContiguous((Tensor<float>)R, nameof(R), opts.Tensor.CopyReporter);
        var bd = B is null ? null : Tensor<float>.RequireContiguous((Tensor<float>)B, nameof(B), opts.Tensor.CopyReporter);
        var hd = initialH is null ? null : Tensor<float>.RequireContiguous((Tensor<float>)initialH, nameof(initialH), opts.Tensor.CopyReporter);
        var cd = initialC is null ? null : Tensor<float>.RequireContiguous((Tensor<float>)initialC, nameof(initialC), opts.Tensor.CopyReporter);
        var pd = P is null ? null : Tensor<float>.RequireContiguous((Tensor<float>)P, nameof(P), opts.Tensor.CopyReporter);
        ReadOnlySpan<float> xs = xd.Buffer.Span;
        ReadOnlySpan<float> ws = wd.Buffer.Span;
        ReadOnlySpan<float> rs = rd.Buffer.Span;
        int yLen = seq * numDirections * batch * hiddenSize;
        int hLen = numDirections * batch * hiddenSize;
        float[] yArr = pool is null ? new float[yLen] : pool.Rent<float>(yLen);
        float[]? yhArr = outputCount > 1 ? (pool is null ? new float[hLen] : pool.Rent<float>(hLen)) : null;
        float[]? ycArr = outputCount > 2 ? (pool is null ? new float[hLen] : pool.Rent<float>(hLen)) : null;
        var xw = new float[4 * hiddenSize];
        var hr = new float[4 * hiddenSize];
        var hv = new float[hiddenSize];
        var cv = new float[hiddenSize];
        int H = hiddenSize;
        for (int d = 0; d < numDirections; d++)
        {
            // The solo reverse direction and the second bidirectional
            // direction both iterate time backwards.
            bool rev = reverse || (numDirections == 2 && d == 1);
            int wDir = d * 4 * H * inputSize;
            int rDir = d * 4 * H * H;
            int bDir = B is null ? 0 : (B.Rank == 2 ? d * 8 * H : 0);
            int pDir = P is null ? 0 : d * 3 * H;
            var fAct = gateF[d];
            var gAct = gateG[d];
            var hAct = gateH[d];
            for (int b = 0; b < batch; b++)
            {
                int limit = lens is null ? seq : Math.Min(lens[b], seq);
                if (hd is null) Array.Clear(hv, 0, H);
                else hd.Buffer.Span.Slice((d * batch + b) * H, H).CopyTo(hv);
                if (cd is null) Array.Clear(cv, 0, H);
                else cd.Buffer.Span.Slice((d * batch + b) * H, H).CopyTo(cv);
                // ORT ignores initial states when the sequence length is zero: final states stay zero.
                if (limit == 0) { Array.Clear(hv, 0, H); Array.Clear(cv, 0, H); }
                for (int s = 0; s < seq; s++)
                {
                    // ORT ReverseSequence reverses only the valid prefix: reverse steps read/write X[limit-1-s]/Y[limit-1-s] for s < limit, with zeros above limit.
                    if (s >= limit)
                    {
                        int zOff = ((s * numDirections + d) * batch + b) * H;
                        Array.Clear(yArr, zOff, H);
                        continue;
                    }
                    int t = rev ? limit - 1 - s : s;
                    int yOff = ((t * numDirections + d) * batch + b) * H;
                    int xOff = (t * batch + b) * inputSize;
                    for (int gh = 0; gh < 4 * H; gh++)
                    {
                        float acc = 0f;
                        int wRow = wDir + gh * inputSize;
                        for (int k = 0; k < inputSize; k++) acc += xs[xOff + k] * ws[wRow + k];
                        xw[gh] = acc;
                    }
                    for (int gh = 0; gh < 4 * H; gh++)
                    {
                        float acc = 0f;
                        int rRow = rDir + gh * H;
                        for (int k = 0; k < H; k++) acc += hv[k] * rs[rRow + k];
                        hr[gh] = acc;
                    }
                    for (int h = 0; h < H; h++)
                    {
                        float wbI = bd is null ? 0f : bd.Buffer.Span[bDir + h];
                        float wbO = bd is null ? 0f : bd.Buffer.Span[bDir + H + h];
                        float wbF = bd is null ? 0f : bd.Buffer.Span[bDir + 2 * H + h];
                        float wbC = bd is null ? 0f : bd.Buffer.Span[bDir + 3 * H + h];
                        float rbI = bd is null ? 0f : bd.Buffer.Span[bDir + 4 * H + h];
                        float rbO = bd is null ? 0f : bd.Buffer.Span[bDir + 5 * H + h];
                        float rbF = bd is null ? 0f : bd.Buffer.Span[bDir + 6 * H + h];
                        float rbC = bd is null ? 0f : bd.Buffer.Span[bDir + 7 * H + h];
                        float iPre = xw[h] + hr[h] + wbI + rbI;
                        float oPre = xw[H + h] + hr[H + h] + wbO + rbO;
                        float fPre = xw[2 * H + h] + hr[2 * H + h] + wbF + rbF;
                        float gPre = xw[3 * H + h] + hr[3 * H + h] + wbC + rbC;
                        if (pd is not null)
                        {
                            ReadOnlySpan<float> ps = pd.Buffer.Span;
                            iPre += ps[pDir + h] * cv[h];
                            fPre += ps[pDir + 2 * H + h] * cv[h];
                        }
                        float fv = fAct(ClipGate(fPre, clip));
                        float gv = gAct(ClipGate(gPre, clip));
                        float iv = fAct(ClipGate(iPre, clip));
                        // Coupled gates tie forget to the input gate
                        // (forget = 1 - input), verified against ORT 1.29.
                        float ff = inputForget ? 1f - iv : fv;
                        float cNew = ff * cv[h] + iv * gv;
                        float oo = oPre;
                        if (pd is not null) oo += pd.Buffer.Span[pDir + H + h] * cNew;
                        float hNew = fAct(ClipGate(oo, clip)) * hAct(cNew);
                        cv[h] = cNew;
                        hv[h] = hNew;
                        yArr[yOff + h] = hNew;
                    }
                }
                if (yhArr is not null) Array.Copy(hv, 0, yhArr, (d * batch + b) * H, H);
                if (ycArr is not null) Array.Copy(cv, 0, ycArr, (d * batch + b) * H, H);
            }
        }
        var outs = new ITensor[outputCount];
        if (outputCount > 0) outs[0] = new DenseTensor<float>(new Memory<float>(yArr), new[] { seq, numDirections, batch, H });
        if (outputCount > 1 && yhArr is not null) outs[1] = new DenseTensor<float>(new Memory<float>(yhArr), new[] { numDirections, batch, H });
        if (outputCount > 2 && ycArr is not null) outs[2] = new DenseTensor<float>(new Memory<float>(ycArr), new[] { numDirections, batch, H });
        return Success(op, outs);
    }

    static float ClipGate(float v, float? clip) =>
        clip.HasValue ? Math.Clamp(v, -clip.Value, clip.Value) : v;

    static float LstmDefaultAlpha(string name) => name.ToLowerInvariant() switch { "hardsigmoid" => 0.2f, "leakyrelu" => 0.01f, "elu" => 1f, _ => 0f };
    static float LstmDefaultBeta(string name) => name.ToLowerInvariant() switch { "hardsigmoid" => 0.5f, _ => 0f };
    static Func<float, float>? LstmActivation(string name, float alpha, float beta) =>
        name.ToLowerInvariant() switch
        {
            "sigmoid" => v => 1f / (1f + MathF.Exp(-v)),
            "tanh" => MathF.Tanh,
            "relu" => v => v > 0f ? v : 0f,
            "affine" => v => alpha * v + beta,
            "leakyrelu" => v => v >= 0f ? v : alpha * v,
            "thresholdedrelu" => v => v > alpha ? v : 0f,
            "scaledtanh" => v => alpha * MathF.Tanh(beta * v),
            "hardsigmoid" => v => Math.Min(1f, Math.Max(0f, alpha * v + beta)),
            "elu" => v => v >= 0f ? v : alpha * (MathF.Exp(v) - 1f),
            "softsign" => v => v / (1f + MathF.Abs(v)),
            "softplus" => v => MathF.Log(MathF.Exp(v) + 1f),
            _ => null,
        };
}
