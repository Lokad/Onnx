"""Constrain arithmetic changes to EpilogueRange and the non-timed range census."""
HELPERS='    delegate bool RangeGuard(ReadOnlySpan<float> values);\n\n    static int RangeCase(RangeGuard check, float[] values, int offset, int length)\n    {\n        var span = values.AsSpan(offset, length);\n        bool expected = true;\n        foreach (float value in span) if (MathF.Abs(value) > float.MaxValue / 4) expected = false;\n        string before = Hash(values);\n        Require(check(span) == expected, "range guard scalar agreement");\n        Require(Hash(values) == before, "range guard input mutation");\n        return 1;\n    }\n\n    static int RangeGuards()\n    {\n        var method = typeof(ConvBlockedSpatial).GetMethod("EpilogueRange",\n            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!;\n        var check = method.CreateDelegate<RangeGuard>();\n        const float limit = float.MaxValue / 4;\n        float below = MathF.BitDecrement(limit), above = MathF.BitIncrement(limit);\n        float[] cases = { 0f, BitConverter.Int32BitsToSingle(int.MinValue), float.Epsilon, -float.Epsilon,\n            below, limit, above, -below, -limit, -above, float.MaxValue, -float.MaxValue,\n            float.PositiveInfinity, float.NegativeInfinity, float.NaN };\n        int count = 0;\n        foreach (int length in new[] { 0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65 })\n        foreach (int offset in new[] { 0, 1, 3 })\n        {\n            // Disallowed magnitude outside the span exposes a vector overread on all-zero baselines.\n            var values = Enumerable.Repeat(float.MaxValue, offset + length + 17).ToArray();\n            values.AsSpan(offset, length).Clear();\n            count += RangeCase(check, values, offset, length);\n            foreach (float value in cases)\n            for (int index = 0; index < length; index++)\n            {\n                values[offset + index] = value;\n                count += RangeCase(check, values, offset, length);\n                values[offset + index] = 0f;\n            }\n        }\n        Require(count == 10566, "range guard census");\n        return count;\n    }\n\n'

def method(text,signature):
    start=text.index(signature);opening=text.index('{',start);depth=1;end=opening+1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return start,end

def check(new,old):
    a=(new/'Winograd.cs').read_text();b=(old/'Winograd.cs').read_text()
    sa,ea=method(a,'    static bool EpilogueRange(');sb,eb=method(b,'    static bool EpilogueRange(')
    assert a[:sa]==b[:sb] and a[ea:]==b[eb:]
    driver=(new/'Driver.cs').read_text();assert driver.count(HELPERS)==1
    driver=driver.replace(HELPERS,'').replace('        int rangeChecks = mode == "raw" ? RangeGuards() : 0;\n','').replace('rows, refusals, contracts, rangeChecks, noPerformanceMeasurement','rows, refusals, contracts, noPerformanceMeasurement')
    assert driver==(old/'Driver.cs').read_text()
