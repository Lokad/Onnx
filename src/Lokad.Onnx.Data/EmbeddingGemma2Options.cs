namespace Lokad.Onnx;

using System;

/// <summary>Bounds for local FP32 EmbeddingGemma 2 inference.</summary>
public sealed record EmbeddingGemma2Options
{
    /// <summary>Maximum tokens per input, including the task prefix and BOS/EOS.</summary>
    public int MaximumTokens { get; init; } = 512;
    /// <summary>Maximum rows per graph evaluation. Requests may span multiple evaluations.</summary>
    public int MaximumBatchSize { get; init; } = 1;
    /// <summary>Maximum batch size times padded sequence length per graph evaluation.</summary>
    public int MaximumPaddedTokens { get; init; } = 512;
    /// <summary>Bound on retained packed CPU weights; original model weights are loaded separately.</summary>
    public long PreparedWeightBudgetBytes { get; init; } = 576L * 1024 * 1024;

    internal void Validate()
    {
        if (MaximumTokens < 2 || MaximumTokens > 8192) throw new ArgumentOutOfRangeException(nameof(MaximumTokens), "Expected 2..8192 tokens.");
        if (MaximumBatchSize < 1) throw new ArgumentOutOfRangeException(nameof(MaximumBatchSize));
        if (MaximumPaddedTokens < MaximumTokens) throw new ArgumentOutOfRangeException(nameof(MaximumPaddedTokens), "Must accommodate one maximum-length input.");
        if (PreparedWeightBudgetBytes < 0) throw new ArgumentOutOfRangeException(nameof(PreparedWeightBudgetBytes));
    }
}
