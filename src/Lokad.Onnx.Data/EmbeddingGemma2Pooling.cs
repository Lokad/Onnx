namespace Lokad.Onnx;

using System;
using System.IO;
using System.Threading;

internal static class EmbeddingGemma2Pooling
{
    internal static void ValidateDimensions(int dimensions)
    {
        if (dimensions is not (128 or 256 or 512 or 768))
            throw new ArgumentOutOfRangeException(nameof(dimensions), "Expected 128, 256, 512, or 768 dimensions.");
    }

    internal static float[][] Pool(Tensor<float> hidden, Tensor<long> mask, int dimensions, CancellationToken cancellation)
    {
        ValidateDimensions(dimensions);
        cancellation.ThrowIfCancellationRequested();
        if (hidden.Rank != 3 || hidden.Dimensions[2] != 768 || mask.Rank != 2 ||
            mask.Dimensions[0] != hidden.Dimensions[0] || mask.Dimensions[1] != hidden.Dimensions[1])
            throw new InvalidDataException("Expected [batch,sequence,768] hidden states and a matching attention mask.");
        int batch = hidden.Dimensions[0], length = hidden.Dimensions[1];
        var result = new float[batch][];
        for (int b = 0; b < batch; b++)
        {
            cancellation.ThrowIfCancellationRequested();
            var vector = new float[dimensions];
            int count = 0;
            for (int t = 0; t < length; t++)
            {
                long valid = mask.GetValue(b * length + t);
                if (valid == 0) continue;
                if (valid != 1) throw new InvalidDataException("Attention mask must contain only 0 or 1.");
                count++;
                for (int d = 0; d < dimensions; d++) vector[d] += hidden.GetValue((b * length + t) * 768 + d);
            }
            if (count == 0) throw new InvalidDataException("No tokens to pool.");
            float squaredNorm = 0;
            for (int d = 0; d < dimensions; d++)
            {
                float value = vector[d] /= count;
                squaredNorm += value * value;
            }
            float norm = MathF.Sqrt(squaredNorm);
            if (!float.IsFinite(norm) || norm <= 1e-12f) throw new InvalidDataException("Nonfinite or zero-norm embedding.");
            for (int d = 0; d < dimensions; d++) vector[d] /= norm;
            result[b] = vector;
        }
        return result;
    }
}
