namespace Lokad.Onnx;

using System;
using System.IO;
using System.Linq;
using System.Threading;

// The weighted statistics follow pyannote.audio StatsPool (CNRS, MIT licensed).
// Keep float intermediates: widening the entire formula changes nearly degenerate masks.
internal static class WeSpeakerPooling
{
    internal static DenseTensor<float> Pool(Tensor<float> encoded, float[] weights,
        bool weighted, CancellationToken cancellation) => PoolCore(encoded, weights, weighted, false, cancellation);

    // The Community-1 pipeline uses native epsilon-regularized statistics even for sparse masks.
    // This is distinct from the public single-vector API's minimum-frame policy.
    internal static DenseTensor<float> PoolPipeline(Tensor<float> encoded, float[] weights,
        CancellationToken cancellation) => PoolCore(encoded, weights, true, true, cancellation);

    static DenseTensor<float> PoolCore(Tensor<float> encoded, float[] weights,
        bool weighted, bool sparseMasks, CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();
        if (encoded.Rank != 3 || encoded.Dimensions[0] != 1 || encoded.Dimensions[2] != weights.Length)
            throw new InvalidDataException("Unexpected WeSpeaker backbone shape.");
        int channels = encoded.Dimensions[1], frames = weights.Length, positive = 0;
        double sum = 0, squares = 0;
        foreach (float weight in weights)
        {
            if (!float.IsFinite(weight) || weight < 0 || weight > 1 || !weighted && weight != 1)
                throw new ArgumentException("Invalid WeSpeaker frame weights.", nameof(weights));
            if (weight > 0) positive++;
            sum += weight;
            squares += (float)(weight * weight);
        }
        if (!sparseMasks && positive < 2) throw new ArgumentException("At least two positive frame weights are required.", nameof(weights));
        float v1 = (float)sum;
        if (weighted) v1 += 1e-8f;
        float denominator = weighted ? v1 - (float)squares / v1 + 1e-8f : frames - 1;
        if (!(denominator > 0)) throw new InvalidDataException("WeSpeaker variance has no positive denominator.");
        float[] values = encoded.ToArray();
        foreach (float value in values)
            if (!float.IsFinite(value)) throw new InvalidDataException("Nonfinite WeSpeaker backbone output.");
        var result = new DenseTensor<float>(new[] { 1, channels * 2 });
        var output = result.Buffer.Span;
        for (int channel = 0; channel < channels; channel++)
        {
            cancellation.ThrowIfCancellationRequested();
            double total = 0;
            for (int t = 0; t < frames; t++) total += (float)(values[channel * frames + t] * weights[t]);
            float mean = (float)total / v1;
            double variance = 0;
            for (int t = 0; t < frames; t++)
            {
                float difference = values[channel * frames + t] - mean;
                variance += (float)((float)(difference * difference) * weights[t]);
            }
            float deviation = MathF.Sqrt((float)variance / denominator);
            if (!float.IsFinite(mean) || !float.IsFinite(deviation))
                throw new InvalidDataException("Nonfinite WeSpeaker statistics.");
            output[channel] = mean;
            output[channel + channels] = deviation;
        }
        return result;
    }
}
