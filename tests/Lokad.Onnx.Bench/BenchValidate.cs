namespace Lokad.Onnx.Bench;

using System;

// Shared benchmark validation: every timed pairing is proven element by
// element before timing. Each breach throws a descriptive error instead of
// timing incorrect results.
internal static class BenchValidate
{
    public static void RequireFinite(string what, float[] values)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i]))
                throw new InvalidOperationException(what + ": non-finite value " + values[i] + " at index " + i + ".");
        }
    }

    public static void RequireSameDims(string what, int[] expectedDims, int[] actualDims)
    {
        if (expectedDims.Length != actualDims.Length)
            throw new InvalidOperationException(what + ": rank mismatch (expected "
                + string.Join("x", expectedDims) + ", actual " + string.Join("x", actualDims) + ").");
        for (int i = 0; i < expectedDims.Length; i++)
        {
            if (expectedDims[i] != actualDims[i])
                throw new InvalidOperationException(what + ": shape mismatch (expected "
                    + string.Join("x", expectedDims) + ", actual " + string.Join("x", actualDims) + ").");
        }
    }

    public static double MaxRelDiff(float[] expected, float[] actual)
    {
        if (expected.Length != actual.Length)
            throw new InvalidOperationException("length mismatch (expected "
                + expected.Length + ", actual " + actual.Length + ").");
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double rel = Math.Abs((double)expected[i] - actual[i]) / (1.0 + Math.Abs((double)expected[i]));
            if (rel > worst) worst = rel;
        }
        return worst;
    }

    public static double RequireAgreement(string what, int[] expectedDims, float[] expected, int[] actualDims, float[] actual, double tolerance)
    {
        RequireSameDims(what, expectedDims, actualDims);
        RequireFinite(what + " expected", expected);
        RequireFinite(what + " actual", actual);
        double worst = MaxRelDiff(expected, actual);
        if (worst > tolerance)
            throw new InvalidOperationException(what + ": outputs diverge (max rel diff " + worst.ToString("E2") + ").");
        return worst;
    }
}
