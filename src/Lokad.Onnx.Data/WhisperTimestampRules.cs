namespace Lokad.Onnx;

using System;
using System.Collections.Generic;

// Timestamp grammar follows OpenAI Whisper ApplyTimestampRules at
// 86098128c0b4f24f0e2aa2994de830614b474227 (MIT). The probability-mass
// comparison is evaluated directly in double log space, without float softmax.
internal static class WhisperTimestampRules
{
    internal const int Begin = 50365;
    internal const int SamplesPerTimestamp = 320;

    internal static void Apply(Span<float> logits, IReadOnlyList<int> tokens, int end)
    {
        logits[Begin - 1] = float.NegativeInfinity;
        bool lastTimestamp = tokens.Count > 0 && tokens[tokens.Count - 1] >= Begin;
        bool previousTimestamp = tokens.Count < 2 || tokens[tokens.Count - 2] >= Begin;
        if (lastTimestamp)
        {
            if (previousTimestamp) logits.Slice(Begin).Fill(float.NegativeInfinity);
            else logits.Slice(0, end).Fill(float.NegativeInfinity);
        }
        for (int i = tokens.Count - 1; i >= 0; i--)
            if (tokens[i] >= Begin)
            {
                int minimum = tokens[i] + (lastTimestamp && !previousTimestamp ? 0 : 1);
                logits.Slice(Begin, minimum - Begin).Fill(float.NegativeInfinity);
                break;
            }
        if (tokens.Count == 0)
        {
            logits.Slice(0, Begin).Fill(float.NegativeInfinity);
            logits.Slice(Begin + 51).Fill(float.NegativeInfinity); // At most one second initially.
        }
        float maximumTimestamp = float.NegativeInfinity, maximumText = float.NegativeInfinity;
        for (int i = 0; i < Begin; i++) maximumText = Math.Max(maximumText, logits[i]);
        for (int i = Begin; i < logits.Length; i++) maximumTimestamp = Math.Max(maximumTimestamp, logits[i]);
        if (float.IsNegativeInfinity(maximumTimestamp)) return;
        double mass = 0;
        for (int i = Begin; i < logits.Length; i++) mass += Math.Exp((double)logits[i] - maximumTimestamp);
        if (maximumTimestamp + Math.Log(mass) > maximumText)
            logits.Slice(0, Begin).Fill(float.NegativeInfinity);
    }
}
