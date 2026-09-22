"""Insert timestamp observations only; reject missing or ambiguous anchors."""
import difflib


def once(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after, 1)


def transform(source):
    start = source.index('    internal static bool Execute(')
    end = source.index('    static bool Finite(', start)
    execute = source[start:end]
    execute = once(execute, '        Geometry(c, m, h, w, stride, lanes);',
        '        Phase.Current.ExecuteStart = System.Diagnostics.Stopwatch.GetTimestamp();\n'
        '        Phase.Current.Executions++;\n        Geometry(c, m, h, w, stride, lanes);')
    execute = once(execute, '            // Preserve exact selected non-finite payload/epilogue behavior.',
        '            Phase.Current.Nonfinite = true;\n            // Preserve exact selected non-finite payload/epilogue behavior.')
    execute = once(execute, '        PackInput(input, packedInput, c, h, w, lanes);',
        '        Phase.Current.Validated = System.Diagnostics.Stopwatch.GetTimestamp();\n'
        '        PackInput(input, packedInput, c, h, w, lanes);\n'
        '        Phase.Current.Packed = System.Diagnostics.Stopwatch.GetTimestamp();')
    execute = once(execute, '        for (int channel = 0; channel < m; channel++)',
        '        Phase.Current.Multiplied = System.Diagnostics.Stopwatch.GetTimestamp();\n'
        '        for (int channel = 0; channel < m; channel++)')
    execute = once(execute, '        return true;',
        '        Phase.Current.Epilogued = System.Diagnostics.Stopwatch.GetTimestamp();\n        return true;')
    source_with_execute = source[:start] + execute + source[end:]
    start = source_with_execute.index('    internal static float[] Run(')
    run = source_with_execute[start:]
    run = once(run, '        Geometry(c, m, h, w, stride, lanes);',
        '        long phaseStart = System.Diagnostics.Stopwatch.GetTimestamp();\n        Geometry(c, m, h, w, stride, lanes);')
    run = once(run, '        try { Execute(input, prepared, bias, residual, result, packedInput, packedOutput, c, m, h, w, stride, lanes, relu); }',
        '        Phase.Current = new Phase.Record { RunStart = phaseStart, Allocated = System.Diagnostics.Stopwatch.GetTimestamp(),\n'
        '            Thread = Environment.CurrentManagedThreadId, Sequence = ++Phase.Sequence };\n'
        '        try { Execute(input, prepared, bias, residual, result, packedInput, packedOutput, c, m, h, w, stride, lanes, relu);\n'
        '            Phase.Current.Executed = System.Diagnostics.Stopwatch.GetTimestamp(); }')
    run = once(run, '        finally { ArrayPool<float>.Shared.Return(packedInput); ArrayPool<float>.Shared.Return(packedOutput); }',
        '        finally { ArrayPool<float>.Shared.Return(packedInput); ArrayPool<float>.Shared.Return(packedOutput);\n'
        '            Phase.Current.Returned = System.Diagnostics.Stopwatch.GetTimestamp(); }')
    result = source_with_execute[:start] + run
    diff = ''.join(difflib.unified_diff(source.splitlines(True), result.splitlines(True), fromfile='qualified/BlockedSpatial.cs', tofile='observed/BlockedSpatial.cs'))
    return result, diff
